"""
Kevin Patel
"""
import sys
import os
import logging
from collections import OrderedDict
from inspect import getfullargspec

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from common_util import is_type, assert_has_all_attr, is_valid, is_type, isnt, dict_flatten, pairwise, np_at_least_nd, np_assert_identical_len_dim
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_LOSS_MAPPING, PYTORCH_OPT_MAPPING, PYTORCH_SCH_MAPPING
from model.train_util import pd_to_np_tvt, get_dataloader
from model.metrics_util import SimulatedReturn
from model.model_util import OutputBlock


class GenericModel(pl.LightningModule):
	"""
	Generic Pytorch Lightning Wrapper.

	Training Hyperparameters:
		window_size (int): window size to use (number of observations in the last dimension of the input tensor)
		feat_dim (int): dimension of resulting feature tensor, if 'None' doesn't reshape
		epochs (int): number training epochs
		batch_size (int): batch (or batch window) size
		batch_step_size (int): batch window step size.
			if this is None DataLoader uses its own default sampler,
			otherwise WindowBatchSampler is used as batch_sampler
		train_shuffle (bool): whether or not to shuffle the order of the training batches
		loss (str): name of the loss function to use,
			'clf' (classsifer) and 'reg' (regressor) are generic 'dummy losses' that only affect how
			the labels/targets are preprocessed (look at model.train_util.py)
		opt (dict): pytorch optimizer settings
			name (str): name of optimizer to use
			kwargs (dict): any keyword arguments to the optimizer constructor
		sch (dict): pytorch scheduler settings
			name (str): name of scheduler to use
			kwargs (dict): any keyword arguments to the scheduler constructor
		num_workers (int>=0): DataLoader option - number cpu workers to attach
		pin_memory (bool): DataLoader option - whether to pin memory to gpu
	"""
	def __init__(self, model_fn, m_params, t_params, data, class_weights=None):
		"""
		Init method

		Args:
			model_fn (function): pytorch model callback
			m_params (dict): dictionary of model hyperparameters
			t_params (dict): dictionary of training hyperparameters
			data (tuple): tuple of pd.DataFrames
			class_weights (dict): class weighting scheme
		"""
		super().__init__()
		self.name = f'{self._get_name()}_{model_fn.__name__}'
		self.m_params, self.t_params = m_params, t_params
		self.hparams = dict_flatten({**self.m_params, **self.t_params})	# Pytorch lightning will track/checkpoint parameters saved in hparams instance variable

		for k, v in filter(lambda i: is_type(i[1], np.ndarray, list, tuple), \
			self.hparams.items()):
			self.hparams[k] = torch.tensor(v).flatten()		# Lists/tuples (and any non-torch primitives) must be stored as flat torch tensors to be tracked by PL
		self.hparams['lr'] = self.t_params['opt']['kwargs']['lr']
		loss_fn = PYTORCH_LOSS_MAPPING.get(self.t_params['loss'], None)
		if (is_valid(loss_fn)):
			self.loss = loss_fn() if (isnt(class_weights)) \
				else loss_fn(weight=class_weights)
		else:
			logging.info('no loss function set in pytorch lightning')
		self.ret_fn = SimulatedReturn(return_type='binary_confidence')
		self.__setup_data__(data)
		self.__build_model__(model_fn)
		## if you specify an example input, the summary will show input/output for each layer
		#self.example_input_array = torch.rand(5, 20)

	@classmethod
	def suggest_params(cls, trial=None):
		"""
		suggest training hyperparameters from an optuna trial object
		or return fixed default hyperparameters

		Pytorch recommends not using num_workers > 0 to return CUDA tensors
		because of the subtleties of CUDA multiprocessing, instead pin the
		memory to the GPU for fast data transfer:
		https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
		"""
		if (is_valid(trial)):
			params = {
				'window_size': trial.suggest_int('window_size', 3, 30),
				'feat_dim': None,
				'train_shuffle': False,
				'epochs': trial.suggest_int('epochs', 200, 500),
				'batch_size': trial.suggest_int('batch_size', 128, 512),
				'batch_step_size': None,
				'loss': 'ce',
				'opt': {
					'name': 'adam',
					'kwargs': {
						'lr': trial.suggest_loguniform('lr', 1e-6, 1e-1)
					}
				},
				'prune_trials': True,
				'num_workers': 0,
				'pin_memory': True
			}
		else:
			params = {
				'window_size': 20,
				'feat_dim': None,
				'train_shuffle': False,    
				'epochs': 200,
				'batch_size': 128,
				'batch_step_size': None,
				'loss': 'ce',
				'opt': {
					'name': 'adam',
					'kwargs': {
						'lr': 1e-3
					}
				},
				'prune_trials': True,
				'num_workers': 0,
				'pin_memory': True
			}
		return params

	def forward(self, x):
		"""
		Run input through the model.
		"""
		return self.model(x)

	def calculate_metrics_step(self, losses, y_hat, y, z=None, calc_pfx=''):
		"""
		process metrics during step
		"""
		if (is_valid(z) and is_valid(self.ret_fn)):
			returns = self.ret_fn(y_hat, y, z, kelly=False)
			kelly_returns = self.ret_fn(y_hat, y, z, kelly=True)
			returns_cumsum = returns.cumsum(dim=0)
			kelly_returns_cumsum = kelly_returns.cumsum(dim=0)
			calc_returns = {
				f'{calc_pfx}_ret': returns.sum(),
				f'{calc_pfx}_kret': kelly_returns.sum(),
				f'{calc_pfx}_retmin': returns_cumsum.min(),
				f'{calc_pfx}_kretmin': kelly_returns_cumsum.min(),
				f'{calc_pfx}_retmax': returns_cumsum.max(),
				f'{calc_pfx}_kretmax': kelly_returns_cumsum.max(),
				f'{calc_pfx}_rethigh': torch.abs(z).sum()
			}
		else:
			calc_returns = {}
		calc_metrics = {f'{calc_pfx}_{k}':v for k,v in losses.items()}

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			for k,v in calc_metrics.items():
				calc_metrics[k] = v.unsqueeze(0)
			for k,v in calc_returns.items():
				calc_returns[k] = v.unsqueeze(0)

		return OrderedDict(**calc_metrics, **calc_returns)

	def aggregate_metrics_end(self, outputs):
		"""
		aggregate metrics at the end of validation or test 
		"""
		# if returned a scalar from _step, outputs is a list of tensor scalars
		# we return just the average in this case (if we want)
		# return torch.stack(outputs).mean()
		output_means = {k:0 for k in outputs[0].keys()}
		for output in outputs:
			output_ = {k:v for k,v in output.items()}

			# reduce manually when using dp
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				for k,v in output_.items():
					output_[k] = torch.mean(v)

			for k in output_means.keys():
				output_means[k] += output_[k]

		for k in output_means.keys():
			output_means[k] /= len(outputs)

		out = output_means.copy()
		out['progress_bar'] = output_means
		out['log'] = output_means
		return out

	def forward_metrics_step(self, batch, batch_idx, calc_pfx=''):
		x, y, z = batch
		y_hat_raw = self.forward(x)
		loss_score = self.loss(y_hat_raw, y)
		losses = {
			'loss': loss_score,
			'acc': GenericModel.acc(y_hat_raw, y, loss_score)
		}
		if (self.t_params['loss'] in ('ce',)):
			y_hat = F.softmax(y_hat_raw, dim=1)
			if (self.m_params['label_size'] == 1):
				y_hat = y_hat[:, 1]
		else:
			y_hat = y_hat_raw
		return self.calculate_metrics_step(losses, y_hat=y_hat, y=y, z=z, \
			calc_pfx=calc_pfx)

	def training_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the training loop
		"""
		tqdm_dict = self.forward_metrics_step(batch, batch_idx, calc_pfx='train')
		return OrderedDict({
			'progress_bar': tqdm_dict,
			'log': tqdm_dict,
			'loss': tqdm_dict['train_loss']
		})

	def validation_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the validation loop
		"""
		return self.forward_metrics_step(batch, batch_idx, calc_pfx='val')

	def test_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the test loop
		"""
		return self.forward_metrics_step(batch, batch_idx, calc_pfx='test')

	validation_end = aggregate_metrics_end
	test_end = aggregate_metrics_end

	def configure_optimizers(self):
		"""
		construct and return optimizers
		"""
		opt_fn = PYTORCH_OPT_MAPPING.get(self.t_params['opt']['name'])
		#opt = opt_fn(self.parameters(), **self.t_params['opt']['kwargs'])
		opt = opt_fn(self.parameters(), lr=self.hparams['lr'])
		return opt
		#sch_fn = PYTORCH_SCH_MAPPING.get(self.t_params['sch']['name'])
		#sch = sch_fn(opt, **self.t_params['sch']['kwargs'])
		#return [opt], [sch]

	def __build_model__(self, model_fn):
		"""
		Feature observation shape - (Channels, Window, Hours or Window Observations)
		"""
		num_channels, num_win, num_win_obs = self.obs_shape
		model_params = {k: v for k, v in self.m_params.items() \
			if (k in getfullargspec(model_fn).args)}
		emb = model_fn(in_shape=(num_channels, num_win*num_win_obs), **model_params)
		self.model = OutputBlock.wrap(emb)	# Appends an OutputBlock if emb.ob_out_shapes exists and is not None

	def __setup_data__(self, data):
		"""
		Set self.flt_{train, val, test} by converting (feature_df, label_df, target_df) to numpy dataframes split across train, val, and test subsets.
		"""
		self.flt_train, self.flt_val, self.flt_test = zip(*map(pd_to_np_tvt, data))
		self.obs_shape = (self.flt_train[0].shape[1], self.t_params['window_size'], self.flt_train[0].shape[-1])	# Feature observation shape - (Channels, Window, Hours or Window Observations)
		shapes = np.asarray(tuple(map(lambda tvt: tuple(map(np.shape, tvt)), (self.flt_train, self.flt_val, self.flt_test))))
		assert all(np.array_equal(a[:, 1:], b[:, 1:]) for a, b in pairwise(shapes)), \
			'feature, label, target shapes must be identical across splits'
		assert all(len(np.unique(mat.T[0, :]))==1 for mat in shapes), \
			'first dimension (N) must be equal in each split for all (f, l, t) tensors'

	# Dataloaders:
	train_dataloader = lambda self: get_dataloader(
		data=self.flt_train,
		loss=self.t_params['loss'],
		window_size=self.t_params['window_size'],
		window_overlap=True,
		feat_dim=self.t_params['feat_dim'],
		batch_size=self.t_params['batch_size'],
		batch_step_size=self.t_params['batch_step_size'],
		batch_shuffle=self.t_params['train_shuffle'],
		num_workers=self.t_params['num_workers'],
		pin_memory=self.t_params['pin_memory'])

	val_dataloader = lambda self: get_dataloader(
		data=self.flt_val,
		loss=self.t_params['loss'],
		window_size=self.t_params['window_size'],
		window_overlap=True,
		feat_dim=self.t_params['feat_dim'],
		batch_size=self.t_params['batch_size'],
		batch_step_size=self.t_params['batch_step_size'],
		num_workers=self.t_params['num_workers'],
		pin_memory=self.t_params['pin_memory'])

	test_dataloader = lambda self: get_dataloader(
		data=self.flt_test,
		loss=self.t_params['loss'],
		window_size=self.t_params['window_size'],
		window_overlap=True,
		feat_dim=self.t_params['feat_dim'],
		batch_size=self.t_params['batch_size'],
		batch_step_size=self.t_params['batch_step_size'],
		num_workers=self.t_params['num_workers'],
		pin_memory=self.t_params['pin_memory'])

	@classmethod
	def acc(cls, y_hat_raw, y, loss_score, on_gpu=True):
		y_hat_dir = torch.argmax(y_hat_raw, dim=1)
		acc_score = torch.sum(y == y_hat_dir).item() / (len(y) * 1.0)
		acc_score = torch.tensor(acc_score)

		if (on_gpu):
			acc_score = acc_score.cuda(loss_score.device.index)
		return acc_score

