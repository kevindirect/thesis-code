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
from model.common import PYTORCH_LOSS_MAPPING
from model.metrics_util import SimulatedReturn
from model.pl_generic import GenericModel


class NP(GenericModel):
	"""
	Neural Process Pytorch Lightning Wrapper.

	TODO:
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
		kelly (bool): whether to use kelly criterion in simulated return
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
		self.ret_fn = SimulatedReturn(return_type='binary_confidence')

	def forward(self, context_x, context_y, target_x, target_y=None, \
		train_mode=False):
		"""
		Run input through the neural process.
		"""
		return self.model(context_x, context_y, target_x, target_y, \
			train_mode=train_mode)

	def forward_metrics_step(self, batch, batch_idx, calc_pfx=''):
		x, y, z = batch
		ctx = self.t_params['batch_size'] // 2 # split batch into context/target
		y_hat, losses = self.forward(x[:ctx], y[:ctx], x[ctx:], y[ctx:], \
			train_mode=calc_pfx=='train')
		return self.calculate_metrics_step(losses, y_hat, y[ctx:], z[ctx:], \
			calc_pfx=calc_pfx)

