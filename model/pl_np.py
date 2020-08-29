"""
Kevin Patel
"""
import sys
import os
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from common_util import is_type, is_valid, isnt
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
		super().__init__(model_fn, m_params, t_params, data, class_weights=None)

	def forward(self, context_x, context_y, target_x, target_y=None, \
		train_mode=False, sample_latent=True):
		"""
		Run input through the neural process.
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
		train_loss = losses['loss']
		train_nll = losses['nll']
		train_kldiv = losses['kldiv']

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			train_loss = train_loss.unsqueeze(0)
			train_nll = train_nll.unsqueeze(0)
			train_kldiv = train_kldiv.unsqueeze(0)

		tqdm_dict = {
			'train_loss': train_loss,
			'train_nll': train_nll,
			'train_kldiv': train_kldiv
		}
		output = OrderedDict({
			'progress_bar': tqdm_dict,
			'log': tqdm_dict,
			'loss': train_loss
		})

		return output # can also return a scalar (train_loss) instead of a dict

	def validation_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the validation loop
		"""
		x, y, z = batch
		ctx = self.t_params['batch_size'] // 2 # split batch into context/target
		y_hat, losses = self.forward(x[:ctx], y[:ctx], x[ctx:], y[ctx:], train_mode=False)
		val_loss = losses['loss']		# TODO change loss calculation
		val_nll = losses['nll']
		val_kldiv = losses['kldiv']

		# # acc
		# labels_hat = torch.argmax(y_hat, dim=1)
		# val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		# val_acc = torch.tensor(val_acc)

		# if (self.on_gpu):
		# 	val_acc = val_acc.cuda(val_loss.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			val_loss = val_loss.unsqueeze(0)
			val_nll = val_nll.unsqueeze(0)
			val_kldiv = val_kldiv.unsqueeze(0)
			# val_acc = val_acc.unsqueeze(0)

		output = OrderedDict({
			'val_loss': val_loss,
			'val_nll': val_nll,
			'val_kldiv': val_kldiv,
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
		val_nll_mean = 0
		val_kldiv_mean = 0
		val_acc_mean = 0
		for output in outputs:
			val_loss = output['val_loss']
			val_nll = output['val_nll']
			val_kldiv = output['val_kldiv']

			# reduce manually when using dp
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				val_loss = torch.mean(val_loss)
				val_nll = torch.mean(val_nll)
				val_kldiv = torch.mean(val_kldiv)
			val_loss_mean += val_loss
			val_nll_mean += val_nll
			val_kldiv_mean += val_kldiv

			# # reduce manually when using dp
			# val_acc = output['val_acc']
			# if (self.trainer.use_dp or self.trainer.use_ddp2):
			# 	val_acc = torch.mean(val_acc)
			# val_acc_mean += val_acc

		val_loss_mean /= len(outputs)
		val_nll_mean /= len(outputs)
		val_kldiv_mean /= len(outputs)
		# val_acc_mean /= len(outputs)

		tqdm_dict = {
			'val_loss': val_loss_mean,
			'val_nll': val_nll_mean,
			'val_kldiv': val_kldiv_mean,
		#	'val_acc': val_acc_mean
		}
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict,
			'val_loss': val_loss_mean,
			'val_nll': val_nll_mean,
			'val_kldiv': val_kldiv_mean
		}
		return result

	def test_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the test loop
		"""
		x, y, z = batch
		ctx = self.t_params['batch_size'] // 2 # split batch into context/target
		y_hat, losses = self.forward(x[:ctx], y[:ctx], x[ctx:], y[ctx:], train_mode=False)
		test_loss = losses['loss']		# TODO change loss calculation
		test_nll = losses['nll']
		test_kldiv = losses['kldiv']

		# # acc
		# labels_hat = torch.argmax(y_hat, dim=1)
		# test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		# test_acc = torch.tensor(test_acc)

		# if (self.on_gpu):
		# 	test_acc = test_acc.cuda(test_loss.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			test_loss = test_loss.unsqueeze(0)
			test_nll = test_nll.unsqueeze(0)
			test_kldiv = test_kldiv.unsqueeze(0)
			# test_acc = test_acc.unsqueeze(0)

		output = OrderedDict({
			'test_loss': test_loss,
			'test_nll': test_nll,
			'test_kldiv': test_kldiv,
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
		test_nll_mean = 0
		test_kldiv_mean = 0
		# test_acc_mean = 0
		for output in outputs:
			test_loss = output['test_loss']
			test_nll = output['test_nll']
			test_kldiv = output['test_kldiv']

			# reduce manually when using dp
			if (self.trainer.use_dp or self.trainer.use_ddp2):
				test_loss = torch.mean(test_loss)
				test_nll = torch.mean(test_nll)
				test_kldiv = torch.mean(test_kldiv)
			test_loss_mean += test_loss
			test_nll_mean += test_nll
			test_kldiv_mean += test_kldiv

			# # reduce manually when using dp
			# test_acc = output['test_acc']
			# if (self.trainer.use_dp or self.trainer.use_ddp2):
			# 	test_acc = torch.mean(test_acc)
			# test_acc_mean += test_acc

		test_loss_mean /= len(outputs)
		test_nll_mean /= len(outputs)
		test_kldiv_mean /= len(outputs)
		# test_acc_mean /= len(outputs)

		tqdm_dict = {
			'test_loss': test_loss_mean,
			'test_nll': test_nll_mean,
			'test_kldiv': test_kldiv_mean,
			# 'test_acc': test_acc_mean
		}
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict,
			'test_loss': test_loss_mean,
			'test_nll': test_nll_mean,
			'test_kldiv': test_kldiv_mean,
		}
		return result

