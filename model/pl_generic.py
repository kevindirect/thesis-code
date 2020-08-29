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
import pytorch_lightning as pl

from common_util import is_type, assert_has_all_attr, is_valid, is_type, isnt, dict_flatten, pairwise, np_at_least_nd, np_assert_identical_len_dim
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_LOSS_MAPPING, PYTORCH_OPT_MAPPING, PYTORCH_SCH_MAPPING
from model.train_util import pd_to_np_tvt, get_dataloader
from model.model_util import OutputLinear


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
		# init superclass
		super().__init__()
		self.hparams = dict_flatten(t_params)				# Pytorch lightning will track/checkpoint parameters saved in hparams instance variable
		for k, v in filter(lambda i: is_type(i[1], np.ndarray, list, tuple), \
			self.hparams.items()):
			self.hparams[k] = torch.tensor(v).flatten()		# Lists/tuples (and any non-torch primitives) must be stored as flat torch tensors to be tracked by PL
		self.m_params, self.t_params = m_params, t_params
		self.hparams['lr'] = self.t_params['opt']['kwargs']['lr']
		loss_fn = PYTORCH_LOSS_MAPPING.get(self.t_params['loss'], None)
		if (is_valid(loss_fn)):
			self.loss = loss_fn() if (isnt(class_weights)) else loss_fn(weight=class_weights)
		self.__setup_data__(data)
		self.__build_model__(model_fn)
		## if you specify an example input, the summary will show input/output for each layer
		#self.example_input_array = torch.rand(5, 20)

	def __build_model__(self, model_fn):
		"""
		"""
		num_channels, num_win, num_win_obs = self.obs_shape						# Feature observation shape - (Channels, Window, Hours / Window Observations)
		emb_params = {k: v for k, v in self.m_params.items() if (k in getfullargspec(model_fn).args)}
		emb = model_fn(in_shape=(num_channels, num_win*num_win_obs), **emb_params)
		if ('out_shapes' in self.m_params.keys() and 'out_init' in self.m_params.keys()):
			self.model = OutputLinear(emb, out_shapes=self.m_params['out_shapes'], init_method=self.m_params['out_init'])
		else:
			self.model = emb

	forward = lambda self, x: self.model(x)

	def training_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the training loop
		"""
		x, y, z = batch
		y_hat = self.forward(x)
		train_loss = self.loss(y_hat, y)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			train_loss = train_loss.unsqueeze(0)

		tqdm_dict = {
			'train_loss': train_loss
		}
		output = OrderedDict({
			'progress_bar': tqdm_dict,
			'log': tqdm_dict,
			'loss': train_loss
		})

		return output # can also return a scalar (loss val) instead of a dict

	def validation_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the validation loop
		"""
		x, y, z = batch
		y_hat = self.forward(x)
		val_loss = self.loss(y_hat, y)

		# acc
		labels_hat = torch.argmax(y_hat, dim=1)
		val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		val_acc = torch.tensor(val_acc)

		if (self.on_gpu):
			val_acc = val_acc.cuda(val_loss.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			val_loss = val_loss.unsqueeze(0)
			val_acc = val_acc.unsqueeze(0)

		output = OrderedDict({
			'val_loss': val_loss,
			'val_acc': val_acc,
		})

		return output # can also return a scalar (loss val) instead of a dict

	def validation_end(self, outputs):
		"""
		Called at the end of validation to aggregate outputs
		:param outputs: list of individual outputs of each validation step
		"""
		# if returned a scalar from validation_step, outputs is a list of tensor scalars
		# we return just the average in this case (if we want)
		# return torch.stack(outputs).mean()

		val_loss_mean = 0
		val_acc_mean = 0
		for output in outputs:
			val_loss = output['val_loss']

			# reduce manually when using dp
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				val_loss = torch.mean(val_loss)
			val_loss_mean += val_loss

			# reduce manually when using dp
			val_acc = output['val_acc']
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				val_acc = torch.mean(val_acc)

			val_acc_mean += val_acc

		val_loss_mean /= len(outputs)
		val_acc_mean /= len(outputs)
		tqdm_dict = {
			'val_loss': val_loss_mean,
			'val_acc': val_acc_mean
		}
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict,
			'val_loss': val_loss_mean
		}
		return result

	def test_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the test loop
		"""
		x, y, z = batch
		y_hat = self.forward(x)
		test_loss= self.loss(y_hat, y)

		# acc
		labels_hat = torch.argmax(y_hat, dim=1)
		test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		test_acc = torch.tensor(test_acc)

		if (self.on_gpu):
			test_acc = test_acc.cuda(test_loss.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			test_loss= test_loss.unsqueeze(0)
			test_acc = test_acc.unsqueeze(0)

		output = OrderedDict({
			'test_loss': test_loss,
			'test_acc': test_acc,
		})

		return output # can also return a scalar (loss test) instead of a dict

	def test_end(self, outputs):
		"""
		Called at the end of test to aggregate outputs
		:param outputs: list of individual outputs of each test step
		"""
		# if returned a scalar from test_step, outputs is a list of tensor scalars
		# we return just the average in this case (if we want)
		# return torch.stack(outputs).mean()

		test_loss_mean = 0
		test_acc_mean = 0
		for output in outputs:
			test_loss = output['test_loss']

			# reduce manually when using dp
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				test_loss = torch.mean(test_loss)
			test_loss_mean += test_loss

			# reduce manually when using dp
			test_acc = output['test_acc']
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				test_acc = torch.mean(test_acc)

			test_acc_mean += test_acc

		test_loss_mean /= len(outputs)
		test_acc_mean /= len(outputs)
		tqdm_dict = {
			'test_loss': test_loss_mean,
			'test_acc': test_acc_mean
		}
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict,
			'test_loss': test_loss_mean
		}
		return result

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

	def __setup_data__(self, data):
		"""
		Set self.flt_{train, val, test} by converting (feature_df, label_df, target_df) to numpy dataframes split across train, val, and test subsets.
		"""
		self.flt_train, self.flt_val, self.flt_test = zip(*map(pd_to_np_tvt, data))
		self.obs_shape = (self.flt_train[0].shape[1], self.t_params['window_size'], self.flt_train[0].shape[-1])	# Feature observation shape - (Channels, Window, Hours or Window Observations)
		shapes = np.asarray(tuple(map(lambda tvt: tuple(map(np.shape, tvt)), (self.flt_train, self.flt_val, self.flt_test))))
		assert all(np.array_equal(a[:, 1:], b[:, 1:]) for a, b in pairwise(shapes)), 'feature, label, target shapes must be identical across splits'
		assert all(len(np.unique(mat.T[0, :]))==1 for mat in shapes), 'first dimension (N) must be identical length in each split for all (feature, label, and target) tensors'

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

	@staticmethod
	def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
		"""
		Parameters you define here will be available to your model through self.params
		"""
		pass

