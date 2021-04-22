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
import torchmetrics as tm

from common_util import is_type, is_valid, isnt
from model.common import WIN_SIZE, PYTORCH_LOSS_MAPPING
from model.pl_generic import GenericModel
from model.metrics_util import SimulatedReturn


class NPModel(GenericModel):
	"""
	Neural Process Pytorch Lightning Wrapper.

	Training Hyperparameters:
		window_size (int): number of observations in the last dimension of the input tensor
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
	def __init__(self, pt_model_fn, m_params, t_params, fobs,
		epoch_metric_types=('train', 'val')):
		"""
		Init method

		Args:
			pt_model_fn (function): neural process pytorch model callback
			m_params (dict): dictionary of model hyperparameters
			t_params (dict): dictionary of training hyperparameters
			fobs (tuple): the shape of a single feature observation,
				this is usually the model input shape
			epoch_metric_types (tuple): which epoch types to init metric objects for
		"""
		super().__init__(pt_model_fn, m_params, t_params, fobs,
			epoch_metric_types=epoch_metric_types)

	def get_context_target(self, batch, train_mode=False):
		"""
		Return context and target points.
		If in regressor mode, y and z are identical.
		"""
		x, y, z = batch
		ctx, tgt = self.t_params['context_size'], self.t_params['context_size'] 

		if (train_mode):
			if (self.t_params['train_sample_context']):
				# XXX causes error with cross attention modules
				ctx = np.random.randint(low=1, high=self.t_params['batch_size']-1, \
					size=None, dtype=int)
				tgt = ctx

			if (self.t_params['train_context_in_target']):
				tgt = 0

		# batch_len = len(x)
		# print(f'context: {ctx}, target: {batch_len-tgt}, len: {batch_len}')

		return x[:ctx], y[:ctx], z[:ctx], x[tgt:], y[tgt:], z[tgt:],

	def forward(self, context_x, context_a, target_x, target_a=None):
		"""
		Run input through the neural process.
		"""
		return self.model(context_x, context_a, target_x, target_y=target_a)

	def forward_step(self, batch, batch_idx, epoch_type='train'):
		"""
		Run forward pass, update step metrics, and return step loss.
		"""
		train_mode = epoch_type == 'train'
		xc, yc, zc, xt, yt, zt = self.get_context_target(batch, train_mode=train_mode)
		if (self.model_type == 'clf'):
			actual_c, actual_t = yc, yt
		elif (self.model_type == 'reg'):
			actual_c, actual_t = zc, zt

		try:
			pred_raw, pred_raw_unc, losses = self.forward(xc, actual_c, xt, \
				target_a=actual_t if (train_mode) else None)
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > forward()\n",
				sys.exc_info()[0], err)
			raise err

		pred = pred_raw.detach().clone().squeeze()
		pred_unc = pred_raw_unc.detach().clone().squeeze()

		if (self.model_type == 'clf'):
			pred = pred.clamp(0.0, 1.0)
			pred_ret = (pred - .5) * 2
		else:
			pred_ret = pred

		# XXX calculate val/test loss here if desired
		# print(pred.cpu())
		# print(pred.dtype)
		# print(pred)
		# print(pred.cpu().argmax(dim=-1).dtype)
		# print(pred.argmax(dim=-1))
		# # print(f'pred: {pred}')
		# # print(f'yt: {yt}')
		if (not train_mode):
			# print('val pred')
			# print(pred[:10])
			# print(pred_unc[:10])
			pass
		else:
			# print('train pred')
			# print(pred[:10])
			# print(pred_unc[:10])
			pass
		#print(torch.linalg.norm(pred_unc))
		# sys.exit()

		try:
			for met in self.epoch_metrics[epoch_type].values():
				met.update(pred.cpu(), actual_t.cpu())
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > met.update()\n",
				sys.exc_info()[0], err)
			print('pred.shape:', pred.shape)
			print('actual_t.shape:', actual_t.shape)
			raise err

		if (is_valid(self.epoch_returns)):
			try:
				for ret in self.epoch_returns[epoch_type].values():
					ret.update(pred_ret.cpu(), zt.cpu())
			except Exception as err:
				print("Error! pl_np.py > NPModel > forward_step() > ret.update()\n",
					sys.exc_info()[0], err)
				print('pred_ret.shape:', pred_ret.shape)
				print('zt.shape:', zt.shape)
				raise err

		return {'loss': losses and losses['loss']} 

	@classmethod
	def suggest_params(cls, trial=None, num_classes=2):
		"""
		suggest training hyperparameters from an optuna trial object
		or return fixed default hyperparameters
		XXX - call super class method to sample most of the tparams
		"""
		if (is_valid(trial)):
			class_weights = torch.zeros(num_classes, dtype=torch.float32, \
				device='cpu', requires_grad=False)
			if (num_classes == 2):
				class_weights[0] = trial.suggest_float(f'class_weights[0]', 0.40, 0.60, step=.01)
				class_weights[1] = 1.0 - class_weights[0]
			else:
				for i in range(num_classes):
					class_weights[i] = trial.suggest_float(f'class_weights[{i}]', \
						0.40, 0.60, step=.01) #1e-6, 1.0, step=1e-6)
				class_weights.div_(class_weights.sum()) # Vector class weights must sum to 1

			batch_size = trial.suggest_int('batch_size', 128, 512, step=128)

			params = {
				'window_size': WIN_SIZE,
				'feat_dim': None,
				'train_shuffle': False,
				'epochs': trial.suggest_int('epochs', 200, 700, step=10),
				'batch_size': batch_size,
				'batch_step_size': batch_size // 4,
				'context_size': batch_size // 2,
				'train_context_in_target': False,
				'train_sample_context': True,
				'loss': 'clf',
				'class_weights': class_weights,
				'opt': {
					'name': 'adam',
					'kwargs': {
						'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True)
					}
				},
				'num_workers': 0,
				'pin_memory': True
			}
		else:
			batch_size = 256
			params = {
				'window_size': WIN_SIZE,
				'feat_dim': None,
				'train_shuffle': False,
				'epochs': 400,
				'batch_size': batch_size,
				'batch_step_size': batch_size // 4,
				'context_size': batch_size // 2,
				'train_context_in_target': False,
				'train_sample_context': True,
				'loss': 'clf',
				'class_weights': None,
				'opt': {
					'name': 'adam',
					'kwargs': {
						'lr': 1e-6
					}
				},
				'num_workers': 0,
				'pin_memory': True
			}
		return params

