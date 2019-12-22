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

from common_util import is_type, assert_has_all_attr, is_valid, isnt, np_at_least_nd, np_assert_identical_len_dim
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_LOSS_MAPPING, PYTORCH_OPT_MAPPING, PYTORCH_SCH_MAPPING
from model.preproc_util import temporal_preproc
from model.train_util import pd_get_np_tvt, batchify
from model.model_util import TemporalConvNet, Classifier


class TCNModel(pl.LightningModule):
	"""
	Top level Temporal CNN Classifer.

	Model Hyperparameters:
		window_size (int): window size to use (number of observations in the last dimension of the input tensor)
		num_blocks (int): number of residual blocks, each block consists of a tcn network and residual connection
		block_shapes (list * list): shape of cnn layers in each block, or individual shape per block in sequence (in units of observation size)
		block_act (str): activation function of each layer in each block
		out_act (str): output activation of each block
		kernel_sizes (list): list of CNN kernel sizes, must be the same length as block_shapes
		dilation_index ('global'|'block'): what index to make each layer dilation a function of
		global_dropout (float): dropout probability of an element to be zeroed for any layer not in no_dropout
		no_dropout (list): list of global layer indices to disable dropout on

	Training Hyperparameters:
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
		self.m_params, self.t_params  = m_params, t_params
		loss_fn = PYTORCH_LOSS_MAPPING.get(self.t_params['loss'])
		self.loss = loss_fn() if (isnt(class_weights)) else loss_fn(weight=class_weights)
		## if you specify an example input, the summary will show input/output for each layer
		self.example_input_array = torch.rand(5, 20)
		self.__setup_data__(data)
		self.__build_model__()

	def __build_model__(self):
		"""
		TCN Based Network
		"""
		num_cols, obs_size = self.feat_shape[-2:]
		history_size = obs_size * self.m_params['window_size']
		scaled_bs = obs_size * np.array(self.m_params['block_shapes'])			# Scale topology by the observation size
		clipped_bs = np.clip(scaled_bs, a_min=1, a_max=None).astype(int).tolist()	# Make sure layer outputs >= 1
		#scaled_ks = obs_size * np.array(self.m_params['kernel_sizes'])

		tcn = TemporalConvNet(
			in_shape=(num_cols, history_size),
			num_blocks=self.m_params['num_blocks'],
			block_shapes=clipped_bs,
			block_act=self.m_params['block_act'],
			out_act=self.m_params['out_act'],
			kernel_sizes=self.m_params['kernel_sizes'],
			dilation_index=self.m_params['dilation_index'],
			global_dropout=self.m_params['global_dropout'],
			no_dropout=self.m_params['no_dropout'])
		self.clf = Classifier(tcn, out_shape=1)

	def forward(self, x):
		"""
		Run input through the model.
		"""
		return self.clf(x)

	def training_step(self, batch, batch_idx):
		"""
		Lightning calls this inside the training loop
		"""
		x, y = batch[:2]
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
		x, y = batch[:2]
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

	def configure_optimizers(self):
		"""
		construct and return optimizers
		"""
		opt_fn = PYTORCH_OPT_MAPPING.get(self.t_params['opt']['name'])
		opt = opt_fn(self.parameters(), **self.t_params['opt']['kwargs'])
		return opt
		#sch_fn = PYTORCH_SCH_MAPPING.get(self.t_params['sch']['name'])
		#sch = sch_fn(opt, **self.t_params['sch']['kwargs'])
		#return [opt], [sch]

	def __setup_data__(self, data):
		feature_df, label_df, target_df = data
		ftrain, fval, ftest = map(np_at_least_nd, pd_get_np_tvt(feature_df, as_midx=False))
		ltrain, lval, ltest = map(partial(np_at_least_nd, axis=-1), pd_get_np_tvt(label_df, as_midx=False))
		ttrain, tval, ttest = map(partial(np_at_least_nd, axis=-1), pd_get_np_tvt(target_df, as_midx=False))
		self.flt_train = (ftrain, ltrain, ttrain); np_assert_identical_len_dim(*self.flt_train)
		self.flt_val = (fval, lval, tval); np_assert_identical_len_dim(*self.flt_val)
		self.flt_test = (ftest, ltest, ttest); np_assert_identical_len_dim(*self.flt_test)

		self.feat_shape = ftrain.shape # Required to infer input shape of model
		#self.num_unique_labels = max(map(lambda a: len(np.unique(a)), (ltrain, lval, ltest))) # TODO - set a variable for size of output layer automatically based on label_df

	def __preproc__(self, data):
		return temporal_preproc(data, window_size=self.m_params['window_size'])

	@pl.data_loader
	def train_dataloader(self):
		logging.info('train_dataloader called')
		return batchify(self.t_params, self.__preproc__(self.flt_train), False)

	@pl.data_loader
	def val_dataloader(self):
		logging.info('val_dataloader called')
		return batchify(self.t_params, self.__preproc__(self.flt_val), False)

	@pl.data_loader
	def test_dataloader(self):
		logging.info('test_dataloader called')
		return batchify(self.t_params, self.__preproc__(self.flt_test), False)

	@staticmethod
	def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
		"""
		Parameters you define here will be available to your model through self.params
		"""
		pass

