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
from model.pl_generic import GenericModel


class NP(GenericModel):
	"""
	Neural Process Pytorch Lightning Wrapper.

	TODO:
		* override relevant methods
		* add optuna objective
		* how will loss function param be used?
		* optimizer?

	Training Hyperparameters:
		window_size (int): window size to use (number of observations in the last dimension of the input tensor)
		feat_dim (int): dimension of resulting feature tensor, if 'None' doesn't reshape
		epochs (int): number training epochs
		batch_size (int): batch (or batch window) size
		batch_step_size (int): batch window step size.
			if this is None DataLoader uses its own default sampler,
			otherwise WindowBatchSampler is used as batch_sampler
		train_shuffle (bool): whether or not to shuffle the order of the training batches
		loss (str): name of loss function to use
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
			model_fn (function): neural process pytorch model callback
			m_params (dict): dictionary of model hyperparameters
			t_params (dict): dictionary of training hyperparameters
			data (tuple): tuple of pd.DataFrames
			class_weights (dict): class weighting scheme
		"""
		# init superclass
		super(NP, self).__init__(model_fn, m_params, t_params, data, class_weights=None)
		# self.hparams = dict_flatten(t_params)				# Pytorch lightning will track/checkpoint parameters saved in hparams instance variable
		# for k, v in filter(lambda i: is_type(i[1], np.ndarray, list, tuple), \
		# 	self.hparams.items()):
		# 	self.hparams[k] = torch.tensor(v).flatten()		# Lists/tuples (and any non-torch primitives) must be stored as flat torch tensors to be tracked by PL
		# self.m_params, self.t_params = m_params, t_params
		# self.hparams['lr'] = self.t_params['opt']['kwargs']['lr']
		# loss_fn = PYTORCH_LOSS_MAPPING.get(self.t_params['loss'])
		# self.loss = loss_fn() if (isnt(class_weights)) else loss_fn(weight=class_weights)
		# self.__setup_data__(data)
		# self.__build_model__(model_fn)
		## if you specify an example input, the summary will show input/output for each layer
		#self.example_input_array = torch.rand(5, 20)

	# def __build_model__(self, model_fn):
	# 	"""
	# 	"""
	# 	num_channels, num_win, num_win_obs = self.obs_shape						# Feature observation shape - (Channels, Window, Hours / Window Observations)
	# 	emb_params = {k: v for k, v in self.m_params.items() if (k in getfullargspec(model_fn).args)}
	# 	emb = model_fn(in_shape=(num_channels, num_win*num_win_obs), **emb_params)
	# 	if ('out_shapes' in self.m_params.keys() and 'out_init' in self.m_params.keys()):
	# 		self.model = OutputLinear(emb, out_shapes=self.m_params['out_shapes'], init_method=self.m_params['out_init'])
	# 	else:
	# 		self.model = emb

	def forward(self, context_x, context_y, target_x, target_y=None, \
		train_mode=False, sample_latent=True):
		"""
		Run input through the model.
		"""
		return self.model(context_x, context_y, target_x, target_y, \
			train_mode=train_mode, sample_latent=sample_latent)

	def training_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the training loop
		"""
		x, y, z = batch
		ctx = self.t_params['batch_size'] // 2 # split batch into context/target
		y_hat, losses = self.forward(x[:ctx], y[:ctx], x[ctx:], y[ctx:], train_mode=True)
		loss_val = losses['loss']		# TODO change loss calculation

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			loss_val = loss_val.unsqueeze(0)

		tqdm_dict = {'train_loss': loss_val}
		output = OrderedDict({
			'loss': loss_val,
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		})

		return output # can also return a scalar (loss val) instead of a dict

	def validation_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the validation loop
		"""
		x, y, z = batch
		ctx = self.t_params['batch_size'] // 2 # split batch into context/target
		y_hat, losses = self.forward(x[:ctx], y[:ctx], x[ctx:], y[ctx:], train_mode=False)
		loss_val = losses['loss']		# TODO change loss calculation

		# # acc
		# labels_hat = torch.argmax(y_hat, dim=1)
		# val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		# val_acc = torch.tensor(val_acc)

		# if (self.on_gpu):
		# 	val_acc = val_acc.cuda(loss_val.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			loss_val = loss_val.unsqueeze(0)
			# val_acc = val_acc.unsqueeze(0)

		output = OrderedDict({
			'val_loss': loss_val,
			# 'val_acc': val_acc,
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

			# # reduce manually when using dp
			# val_acc = output['val_acc']
			# if (self.trainer.use_dp or self.trainer.use_ddp2):
			# 	val_acc = torch.mean(val_acc)

			# val_acc_mean += val_acc

		val_loss_mean /= len(outputs)
		# val_acc_mean /= len(outputs)
		tqdm_dict = {
			'val_loss': val_loss_mean,
		#	'val_acc': val_acc_mean
		}
		result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
		return result

	def test_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the test loop
		"""
		x, y, z = batch
		ctx = self.t_params['batch_size'] // 2 # split batch into context/target
		y_hat, losses = self.forward(x[:ctx], y[:ctx], x[ctx:], y[ctx:], train_mode=False)
		loss_val = losses['loss']		# TODO change loss calculation

		# # acc
		# labels_hat = torch.argmax(y_hat, dim=1)
		# test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		# test_acc = torch.tensor(test_acc)

		# if (self.on_gpu):
		# 	test_acc = test_acc.cuda(loss_test.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			loss_test = loss_test.unsqueeze(0)
			# test_acc = test_acc.unsqueeze(0)

		output = OrderedDict({
			'test_loss': loss_test,
			# 'test_acc': test_acc,
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
		# test_acc_mean = 0
		for output in outputs:
			test_loss = output['test_loss']

			# reduce manually when using dp
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				test_loss = torch.mean(test_loss)
			test_loss_mean += test_loss

			# # reduce manually when using dp
			# test_acc = output['test_acc']
			# if (self.trainer.use_dp or self.trainer.use_ddp2):
			# 	test_acc = torch.mean(test_acc)

			# test_acc_mean += test_acc

		test_loss_mean /= len(outputs)
		# test_acc_mean /= len(outputs)
		tqdm_dict = {
			'test_loss': test_loss_mean,
			# 'test_acc': test_acc_mean
		}
		result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_loss': test_loss_mean}
		return result

