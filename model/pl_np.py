"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

	def __init_loss_fn__(self):
		if (is_valid(loss_fn := PYTORCH_LOSS_MAPPING.get(self.t_params['loss'], None))):
			if (is_valid(self.t_params['class_weights'])):
				self.loss = loss_fn(reduction='none', \
					weight=self.t_params['class_weights'])
			else:
				self.loss = loss_fn(reduction='none')
		else:
			logging.info('no loss function set in pytorch lightning')

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
		Run input through model and return output.
		Use at inference time only.
		"""
		raise NotImplementedError()

	def forward_step(self, batch, batch_idx, epoch_type='train'):
		"""
		Run forward pass, calculate step loss, and calculate step metrics.
		"""
		train_mode = epoch_type == 'train'
		xc, yc, zc, xt, yt, zt = self.get_context_target(batch, train_mode=train_mode)
		if (self.model_type == 'clf'):
			actual_c, actual_t = yc, yt
		elif (self.model_type == 'reg'):
			actual_c, actual_t = zc, zt

		try:
			prior_dist, post_dist, out_dist = self.model(xc, actual_c, xt, \
				target_y=actual_t if (train_mode) else None)
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > model()\n",
				sys.exc_info()[0], err)
			print(f'{train_mode=}')
			print(f'{xc.shape=}')
			print(f'{xt.shape=}')
			print(f'{actual_c.shape=}')
			print(f'{actual_t.shape=}')
			print(f'{zc.shape=}')
			print(f'{zt.shape=}')
			raise err

		if (self.t_params['sample_out'] and train_mode and out_dist.has_rsample):
			pred_t_raw = out_dist.rsample()
		else:
			pred_t_raw = out_dist.mean
		pred_t_raw = pred_t_raw.squeeze()
		pred_t_unc_raw = out_dist.stddev.squeeze()
		#print(torch.linalg.norm(pred_t_unc))

		# Reshape model outputs for later loss, metrics calculations:
		if (self.model_type == 'clf'):
			if (self.t_params['loss'] in ('clf-dnll',)):
				pred_t_loss = out_dist
				pred_t = pred_t_raw.detach().clone().clamp(0.0, 1.0)
				pred_t_ret = (pred_t - .5) * 2
			elif (self.t_params['loss'] in ('clf-ce',)):
				pred_t_loss = F.softmax(pred_t_raw, dim=-1)
				pred_t_conf, pred_t = pred_t_loss.detach().clone() \
					.max(dim=-1, keepdim=False)
				pred_t_dir = pred_t.detach().clone()
				pred_t_dir[pred_t_dir==0] = -1
				pred_t_ret = pred_t_dir * pred_t_conf
			else:
				pred_t_loss = pred_t_raw
				raise NotImplementedError()
		elif (self.model_type == 'reg'):
			if (self.t_params['loss'] in ('reg-dnll',)):
				pred_t_loss = out_dist
			elif (self.t_params['loss'] in ('reg-mae', 'reg-mse')):
				pred_t_loss = pred_t_raw
			pred_t_ret = pred_t = pred_t_raw.detach().clone()

		elbo_loss = None
		if (train_mode):
			if (is_valid(prior_dist) and is_valid(post_dist)):
				kldiv = torch.distributions.kl_divergence(post_dist, prior_dist) \
					.mean(dim=-1, keepdim=True)
			else:
				kldiv = 0

			try:
				model_loss = self.loss(pred_t_loss, actual_t)
			except Exception as err:
				print("Error! pl_np.py > NPModel > forward_step() > loss()\n",
					sys.exc_info()[0], err)
				print(f'{self.loss=}')
				print(f'{pred_t_loss.shape=}')
				print(f'{actual_t.shape=}')
				raise err
			# print('self.loss:', self.loss)
			# print('pred_t_loss.shape:', pred_t_loss.shape)
			# print('pred_t_loss:', pred_t_loss)
			# print('actual_t.shape:', actual_t.shape)
			# print('actual_t:', actual_t)
			# print('kldiv.shape:', kldiv.shape)
			# print('kldiv:', kldiv)
			# print('model_loss.shape:', model_loss.shape)
			# print('model_loss:', model_loss)
			# sys.exit()
			elbo_loss = (kldiv + model_loss).mean()

		for met in self.epoch_metrics[epoch_type].values():
			try:
				met.update(pred_t.cpu(), actual_t.cpu())
			except Exception as err:
				print("Error! pl_np.py > NPModel > forward_step() > met.update()\n",
					sys.exc_info()[0], err)
				print(f'{met=}')
				print(f'{pred_t.shape=}')
				print(f'{actual_t.shape=}')
				raise err

		if (is_valid(self.epoch_returns)):
			for ret in self.epoch_returns[epoch_type].values():
				try:
					ret.update(pred_t_ret.cpu(), zt.cpu())
				except Exception as err:
					print("Error! pl_np.py > NPModel > forward_step() > ret.update()\n",
						sys.exc_info()[0], err)
					print(f'{ret=}')
					print(f'{pred_t_ret.shape=}')
					print(f'{zt.shape=}')
					raise err

		return {'loss': elbo_loss}

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

