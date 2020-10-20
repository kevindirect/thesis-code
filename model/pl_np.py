"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from common_util import is_type, is_valid, isnt
from model.common import PYTORCH_LOSS_MAPPING
from model.metrics_util import SimulatedReturn
from model.pl_generic import GenericModel


class NPModel(GenericModel):
	"""
	Neural Process Pytorch Lightning Wrapper.

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
	def __init__(self, model_fn, m_params, t_params, data, class_weights=None,
		epoch_metric_types=('train', 'val')):
		"""
		Init method

		Args:
			model_fn (function): neural process pytorch model callback
			m_params (dict): dictionary of model hyperparameters
			t_params (dict): dictionary of training hyperparameters
			data (tuple): tuple of pd.DataFrames
			class_weights (dict): class weighting scheme
		"""
		super().__init__(model_fn, m_params, t_params, data, class_weights=None, \
			epoch_metric_types=epoch_metric_types)

	@classmethod
	def suggest_params(cls, trial=None, num_classes=2):
		"""
		suggest training hyperparameters from an optuna trial object
		or return fixed default hyperparameters
		XXX - call super class method to sample most of the tparams
		"""
		if (is_valid(trial)):
			raise Error('look at batch_step_size')
			params = {
				'window_size': trial.suggest_int('window_size', 3, 120),
				'feat_dim': None,
				'train_shuffle': False,
				'epochs': trial.suggest_int('epochs', 200, 700, step=10),
				'batch_size': trial.suggest_int('batch_size', 64, 512, step=8),
				'batch_step_size': trial.suggest_int('batch_step_size', 1, 64),
				'loss': 'clf',
				'class_weights': None,
				'opt': {
					'name': 'adam',
					'kwargs': {
						'lr': trial.suggest_float('lr', 1e-6, 1e-1, log=True)
					}
				},
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
				'batch_step_size': 64,
				'loss': 'clf',
				'class_weights': None,
				'opt': {
					'name': 'adam',
					'kwargs': {
						'lr': 1e-3
					}
				},
				'num_workers': 0,
				'pin_memory': True
			}
		return params

	def forward(self, context_x, context_y, target_x, target_y=None, \
		train_mode=False):
		"""
		Run input through the neural process.
		"""
		return self.model(context_x, context_y, target_x, target_y, \
			train_mode=train_mode)

	def forward_step(self, batch, batch_idx, epoch_type='train'):
		raise NotImplementedError()

	# def forward_metrics_step(self, batch, batch_idx, calc_pfx=''):
	# 	x, y, z = batch
	# 	ctx = self.t_params['batch_size'] // 2 # split batch into context/target
	# 	y_hat, losses = self.forward(x[:ctx], y[:ctx], x[ctx:], y[ctx:], \
	# 		train_mode=calc_pfx=='train')
	# 	return self.calculate_metrics_step(losses, y_hat, y[ctx:], z[ctx:], \
	# 		calc_pfx=calc_pfx)

