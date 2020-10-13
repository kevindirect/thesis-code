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
from pytorch_lightning.metrics.classification import Accuracy#, Fbeta

from common_util import load_df, dump_df, is_type, assert_has_all_attr, is_valid, is_type, isnt, dict_flatten, pairwise, np_at_least_nd, np_assert_identical_len_dim
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
	def __init__(self, model_fn, m_params, t_params, data, class_weights=None,\
		epoch_metric_types=('train', 'val')):
		"""
		Init method

		Args:
			model_fn (function): pytorch model callback
			m_params (dict): dictionary of model hyperparameters
			t_params (dict): dictionary of training hyperparameters
			data (tuple): tuple of pd.DataFrames
			class_weights (dict): class weighting scheme
			epoch_metric_types (tuple): which epoch types to init metric objects for
		"""
		super().__init__()
		self.name = f'{self._get_name()}_{model_fn.__name__}'
		self.m_params, self.t_params = m_params, t_params
		# PL will track/checkpoint parameters saved in hparams instance variable:
		self.hparams = dict_flatten({**self.m_params, **self.t_params})

		# Lists/tuples must be stored as flat torch tensors to be tracked by PL:
		for k, v in filter(lambda i: is_type(i[1], np.ndarray, list, tuple), \
			self.hparams.items()):
			self.hparams[k] = torch.tensor(v).flatten()
		self.hparams['lr'] = self.t_params['opt']['kwargs']['lr']
		loss_fn = PYTORCH_LOSS_MAPPING.get(self.t_params['loss'], None)
		if (is_valid(loss_fn)):
			self.loss = loss_fn() if (isnt(class_weights)) \
				else loss_fn(weight=class_weights)
		else:
			logging.info('no loss function set in pytorch lightning')
		self.__setup_data__(data)
		self.__build_model__(model_fn)

		self.epoch_metrics = {
			epoch_type: {
				'acc': Accuracy(compute_on_step=False),
				# 'fbeta(.5)': Fbeta(num_classes=self.m_params['label_size'], \
				# 	beta=0.5, average='macro', compute_on_step=False)
			}
			for epoch_type in epoch_metric_types
		}
		self.epoch_returns = {
			epoch_type: {
				'cret': SimulatedReturn(use_conf=True, use_kelly=False),
				'kret': SimulatedReturn(use_conf=True, use_kelly=True)
			}
			for epoch_type in epoch_metric_types
		}

		# the summary will show input/output for each layer if example input is set:
		# self.example_input_array = torch.rand(5, 20)

	def configure_optimizers(self):
		"""
		Construct and return optimizers
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
		self.model = OutputBlock.wrap(emb)	# Append OB if is_valid(emb.ob_out_shapes)

	def forward(self, x):
		"""
		Run input through the model.
		"""
		return self.model(x)

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

	def forward_step(self, batch, batch_idx, epoch_type='train'):
		"""
		Run forward pass, update step metrics, and return step loss.
		"""
		x, y, z = batch
		y_hat_raw = self.forward(x)
		step_loss = self.loss(y_hat_raw, y)
		if (self.t_params['loss'] in ('ce',)):
			y_hat = F.softmax(y_hat_raw, dim=1)
			if (self.m_params['label_size'] == 1):
				y_hat = y_hat[:, 1]
		else:
			y_hat = y_hat_raw

		for met in self.epoch_metrics[epoch_type].values():
			met.update(y_hat.cpu(), y.cpu())
		for ret in self.epoch_returns[epoch_type].values():
			ret.update(y_hat.cpu(), y.cpu(), z.cpu())

		return {'loss': step_loss}

	def aggregate_loss_epoch_end(self, outputs, epoch_type='train'):
		"""
		Aggregate step losses into epoch loss and log it.
		"""
		step_losses = [d['loss'].cpu() for d in outputs]
		epoch_loss = torch.mean(torch.stack(step_losses), dim=0)
		# epoch_loss = step_losses[0] if (len(step_losses)==1 and step_losses[0].dim()==0) \
		# 	else torch.mean(torch.stack(step_losses), dim=0)
		self.log('epoch', self.trainer.current_epoch, prog_bar=False, \
			logger=True, on_step=False, on_epoch=True)
		self.log(f'{epoch_type}_loss', epoch_loss, prog_bar=False, \
			logger=True, on_step=False, on_epoch=True)

	def compute_metrics_epoch_end(self, epoch_type='train'):
		"""
		Compute, log, and reset metrics at the end of an epoch.
		"""
		for name, met in self.epoch_metrics[epoch_type].items():
			self.log(f'{epoch_type}_{name}', met.compute(), prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			met.reset()
		for name, ret in self.epoch_returns[epoch_type].items():
			self.log_dict(ret.compute(pfx=epoch_type), prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			ret.reset()

	def training_step(self, batch, batch_idx, epoch_type='train'):
		"""
		Compute and return training step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def training_epoch_end(self, outputs, epoch_type='train'):
		"""
		Aggregate training step losses and step metrics and log them all.
		"""
		self.aggregate_loss_epoch_end(outputs, epoch_type)
		self.compute_metrics_epoch_end(epoch_type)

	def validation_step(self, batch, batch_idx, epoch_type='val'):
		"""
		Compute and return validation step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def validation_epoch_end(self, outputs, epoch_type='val'):
		"""
		Aggregate validation step losses and step metrics and log them all.
		"""
		self.aggregate_loss_epoch_end(outputs, epoch_type)
		self.compute_metrics_epoch_end(epoch_type)

	def test_step(self, batch, batch_idx, epoch_type='test'):
		"""
		Compute and return test step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def test_epoch_end(self, outputs, epoch_type='test'):
		"""
		Aggregate test step losses and step metrics and log them all.
		"""
		self.aggregate_loss_epoch_end(outputs, epoch_type)
		self.compute_metrics_epoch_end(epoch_type)

	# XXX use pl.DataModule instead of this:
	def __setup_data__(self, data):
		"""
		Set self.flt_{train, val, test} by converting (feature_df, label_df, target_df) to numpy dataframes split across train, val, and test subsets.
		"""
		self.flt_train, self.flt_val, self.flt_test = zip(*map(pd_to_np_tvt, data))
		self.obs_shape = (self.flt_train[0].shape[1], self.t_params['window_size'], \
			self.flt_train[0].shape[-1])	# (Channels, Window, Hours or Window Obs)
		shapes = np.asarray(tuple(map(lambda tvt: tuple(map(np.shape, tvt)), \
			(self.flt_train, self.flt_val, self.flt_test))))
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
	def fix_metrics_csv(cls, fname, dir_path):
		"""
		Fix Pytorch Lightning v10 logging rows in the same epoch on
		separate rows.
		"""
		csv_df = load_df(fname, dir_path=dir_path, data_format='csv')
		csv_df = csv_df.groupby('epoch').ffill().dropna(how='any')
		dump_df(csv_df, f'fix_{fname}', dir_path=dir_path, data_format='csv')
		logging.info(f'fixed {fname}')

