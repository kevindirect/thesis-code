"""
Kevin Patel
"""
import sys
import os
import logging
from functools import partial
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from common_util import is_type, assert_has_all_attr, is_valid, is_type, isnt, dict_flatten, pairwise, np_at_least_nd, np_assert_identical_len_dim
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_LOSS_MAPPING, PYTORCH_OPT_MAPPING, PYTORCH_SCH_MAPPING
from model.train_util import pd_to_np_tvt, get_dataloader
from model.model_util import TemporalConvNet, OutputLinear


class TCNModel(pl.LightningModule):
	"""
	Top level Temporal Convolutional Network.

	Model Hyperparameters:
		pad_type (str): padding method to use ('same' or 'full')
		num_blocks (int): number of residual blocks, each block consists of a tcn network and residual connection
		block_channels (list * list): cnn channel sizes in each block, or individual channel sizes per block in sequence
		block_act (str): activation function of each layer in each block
		out_act (str): output activation of each block
		block_init (str): layer weight initialization method of each layer in each block
		out_init (str): layer weight initialization method of each block
		kernel_sizes (list): list of CNN kernel sizes, must be the same length as block_channels
		dilation_index ('global'|'block'): what index to make each layer dilation a function of
		global_dropout (float): dropout probability of an element to be zeroed for any layer not in no_dropout
		no_dropout (list): list of global layer indices to disable dropout on

	Training Hyperparameters:
		window_size (int): window size to use (number of observations in the last dimension of the input tensor)
		flatten_features (bool): whether or not to flatten the feature tensor to (N, C)
			where N is the original first dimension and C is the product of all following dimensions
		epochs (int): number training epochs
		batch_size (int): training batch size
		loss (str): name of loss function to use
		opt (dict): pytorch optimizer settings
			name (str): name of optimizer to use
			kwargs (dict): any keyword arguments to the optimizer constructor
		sch (dict): pytorch scheduler settings
			name (str): name of scheduler to use
			kwargs (dict): any keyword arguments to the scheduler constructor
	"""
	def __init__(self, m_params, t_params, data, class_weights=None):
		"""
		Init method

		Args:
			m_params (dict): dictionary of model (hyper)parameters
			t_params (dict): dictionary of training (hyper)parameters
			data (tuple): tuple of pd.DataFrames
			class_weights (dict): class weighting scheme
		"""
		# init superclass
		super(TCNModel, self).__init__()
		self.hparams = dict_flatten({**t_params, **m_params})		# Pytorch lightning will track/checkpoint parameters saved in hparams instance variable
		for k, v in filter(lambda i: is_type(i[1], np.ndarray, list, tuple), self.hparams.items()):
			self.hparams[k] = torch.tensor(v).flatten()		# Lists/tuples (and any non-torch primitives) must be stored as flat torch tensors to be tracked by PL
		self.m_params, self.t_params = m_params, t_params
		self.hparams['lr'] = self.t_params['opt']['kwargs']['lr']
		loss_fn = PYTORCH_LOSS_MAPPING.get(self.t_params['loss'])
		self.loss = loss_fn() if (isnt(class_weights)) else loss_fn(weight=class_weights)
		## if you specify an example input, the summary will show input/output for each layer
		#self.example_input_array = torch.rand(5, 20)
		self.__setup_data__(data)
		self.__build_model__()

	def __build_model__(self):
		"""
		TCN Based Network
		"""
		num_channels, num_win, num_win_obs = self.obs_shape					# Feature observation shape - (Channels, Window, Hours / Window Observations)
		scaled_bcs = num_win_obs * np.array(self.m_params['block_channels'])			# Scale topology by the observation size
		clipped_bcs = np.clip(scaled_bcs, a_min=1, a_max=None).astype(int).tolist()		# Make sure layer shape dims >= 1
		#scaled_ks = obs_size * np.array(self.m_params['kernel_sizes'])

		tcn = TemporalConvNet(
			in_shape=(num_channels, num_win*num_win_obs),
			pad_type=self.m_params['pad_type'],
			num_blocks=self.m_params['num_blocks'],
			block_channels=clipped_bcs,
			block_act=self.m_params['block_act'],
			out_act=self.m_params['out_act'],
			block_init=self.m_params['block_init'],
			out_init=self.m_params['out_init'],
			kernel_sizes=self.m_params['kernel_sizes'],
			dilation_index=self.m_params['dilation_index'],
			global_dropout=self.m_params['global_dropout'],
			no_dropout=self.m_params['no_dropout'])
		self.clf = OutputLinear(tcn, out_shape=self.m_params['out_shape'], init_method=self.m_params['out_init'])

	def forward(self, x):
		"""
		Run input through the model.
		"""
		return self.clf(x)

	def training_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the training loop
		"""
		x, y, z = batch
		y_hat = self.forward(x)
		loss_val = self.loss(y_hat, y)

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
		y_hat = self.forward(x)
		loss_val = self.loss(y_hat, y)

		# acc
		labels_hat = torch.argmax(y_hat, dim=1)
		val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		val_acc = torch.tensor(val_acc)

		if (self.on_gpu):
			val_acc = val_acc.cuda(loss_val.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			loss_val = loss_val.unsqueeze(0)
			val_acc = val_acc.unsqueeze(0)

		output = OrderedDict({
			'val_loss': loss_val,
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
		tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
		result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
		return result

	def test_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the test loop
		"""
		x, y, z = batch
		y_hat = self.forward(x)
		loss_test = self.loss(y_hat, y)

		# acc
		labels_hat = torch.argmax(y_hat, dim=1)
		test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
		test_acc = torch.tensor(test_acc)

		if (self.on_gpu):
			test_acc = test_acc.cuda(loss_test.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if (self.trainer.use_dp or self.trainer.use_ddp2):
			loss_test = loss_test.unsqueeze(0)
			test_acc = test_acc.unsqueeze(0)

		output = OrderedDict({
			'test_loss': loss_test,
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
		tqdm_dict = {'test_loss': test_loss_mean, 'test_acc': test_acc_mean}
		result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_loss': test_loss_mean}
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
		self.obs_shape = (self.flt_train[0].shape[1], self.t_params['window_size'], self.flt_train[0].shape[-1])	# Feature observation shape - (Channels, Window, Hours / Window Observations)
		shapes = np.asarray(tuple(map(lambda tvt: tuple(map(np.shape, tvt)), (self.flt_train, self.flt_val, self.flt_test))))
		assert all(np.array_equal(a[:, 1:], b[:, 1:]) for a, b in pairwise(shapes)), 'feature, label, target shapes must be identical across splits'
		assert all(len(np.unique(mat.T[0, :]))==1 for mat in shapes), 'first dimension (N) must be identical length in each split for all (feature, label, and target) tensors'

	# Dataloaders:
	train_dataloader = lambda self: get_dataloader(data=self.flt_train, loss=self.t_params['loss'], window_size=self.t_params['window_size'], window_overlap=True, flatten_features=self.t_params['flatten_features'], batch_size=self.t_params['batch_size'], shuffle=False)

	val_dataloader = lambda self: get_dataloader(data=self.flt_val, loss=self.t_params['loss'], window_size=self.t_params['window_size'], window_overlap=True, flatten_features=self.t_params['flatten_features'], batch_size=self.t_params['batch_size'], shuffle=False)

	test_dataloader = lambda self: get_dataloader(data=self.flt_test, loss=self.t_params['loss'], window_size=self.t_params['window_size'], window_overlap=True, flatten_features=self.t_params['flatten_features'], batch_size=self.t_params['batch_size'], shuffle=False)

	@staticmethod
	def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
		"""
		Parameters you define here will be available to your model through self.params
		"""
		pass

