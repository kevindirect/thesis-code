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

from common_util import is_type, is_valid, isnt, pt_resample_values
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
		if (is_valid(cs:=self.t_params['context_size']) and is_valid(ts:=self.t_params['target_size'])):
			assert cs+ts == self.t_params['batch_size'], \
				"context and target sizes must add up to batch_size, \
				use train_target_overlap to add a training overlap"

	def __init_loss_fn__(self, reduction='none'):
		if (is_valid(loss_fn := PYTORCH_LOSS_MAPPING.get(self.t_params['loss'], None))):
			if (is_valid(self.t_params['class_weights'])):
				self.loss = loss_fn(reduction=reduction, \
					weight=self.t_params['class_weights'])
			else:
				self.loss = loss_fn(reduction=reduction)
		else:
			logging.info('no loss function set in pytorch lightning')

	def get_context_target(self, batch, train_mode=False):
		"""
		Return context and target points.
		If in regressor mode, y and z are identical.
		"""
		x, y, z = batch
		ctx = self.t_params['context_size'] or self.t_params['batch_size']//2
		tgt = self.t_params['target_size'] or self.t_params['batch_size'] - ctx

		if (train_mode):
			if (self.t_params['train_sample_context_size']):
				ctx = np.random.randint(low=1, high=self.t_params['batch_size'], \
					size=None, dtype=int)
				tgt = self.t_params['batch_size'] - ctx

			tto = self.t_params['train_target_overlap']
			tend = None
			if (is_type(tto := self.t_params['train_target_overlap'], int)):
				tgt += tto
				if (tto > 0):
					tend = -tto

			if (is_valid(ts := self.t_params['train_resample']) and \
				self.model_type == 'clf'):
				ctx_idx = pt_resample_values(y[:ctx], n=ts, shuffle=True) \
					.to(y.device)
				tgt_idx = pt_resample_values(y[-tgt:tend], n=ts, shuffle=True) \
					.to(y.device)

				return x[:ctx].index_select(dim=0, index=ctx_idx), \
					y[:ctx].index_select(dim=0, index=ctx_idx), \
					z[:ctx].index_select(dim=0, index=ctx_idx), \
					x[-tgt:tend].index_select(dim=0, index=tgt_idx), \
					y[-tgt:tend].index_select(dim=0, index=tgt_idx), \
					z[-tgt:tend].index_select(dim=0, index=tgt_idx)
		else:
			return x[:ctx], y[:ctx], z[:ctx], x[-tgt:], y[-tgt:], z[-tgt:],

	def forward(self, context_x, context_a, target_x, target_a=None, sample_out=False):
		"""
		Run input through model and return output.
		Use at inference time only.
		"""
		try:
			prior_dist, post_dist, out_dist = self.model(xc, context_a, xt, \
				target_y=target_a)
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward() > model()\n",
				sys.exc_info()[0], err)
			print(f'{train_mode=}')
			print(f'{xc.shape=}')
			print(f'{xt.shape=}')
			print(f'{aim_c.shape=}')
			print(f'{aim_t.shape=}')
			print(f'{zc.shape=}')
			print(f'{zt.shape=}')
			raise err

		if (sample_out and out_dist.has_rsample):
			pred_t_raw = out_dist.rsample()
		else:
			pred_t_raw = out_dist.mean

		pred_t_raw = pred_t_raw.squeeze()
		pred_t_unc_raw = out_dist.stddev.squeeze()
		return pred_t_raw, pred_t_unc_raw

	def forward_step(self, batch, batch_idx, epoch_type):
		"""
		Run forward pass, calculate step loss, and calculate step metrics.
		"""
		train_mode = epoch_type == 'train'
		xc, yc, zc, xt, yt, zt = self.get_context_target(batch, train_mode=train_mode)
		# print(f'{xc.shape[0]=} {xt.shape[0]=}')

		if (self.model_type == 'clf'):
			aim_c, aim_t = yc, yt
		elif (self.model_type == 'reg'):
			aim_c, aim_t = zc, zt

		try:
			prior_dist, post_dist, out_dist = self.model(xc, aim_c, xt, \
				target_y=aim_t if (train_mode) else None)
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > model()\n",
				sys.exc_info()[0], err)
			print(f'{train_mode=}')
			print(f'{xc.shape=}')
			print(f'{xt.shape=}')
			print(f'{aim_c.shape=}')
			print(f'{aim_t.shape=}')
			print(f'{zc.shape=}')
			print(f'{zt.shape=}')
			raise err

		if (self.t_params['sample_out'] and train_mode and out_dist.has_rsample):
			pred_t_raw = out_dist.rsample()
		else:
			pred_t_raw = out_dist.mean
		pred_t_raw = pred_t_raw.squeeze()
		pred_t_unc_raw = out_dist.stddev.squeeze()
		# print(torch.linalg.norm(pred_t_unc))
		# print(f'{pred_t_raw.shape=}')
		# print(f'{pred_t_raw=}')
		if (pred_t_raw.ndim == 3):
			# collapse channels if they exist:
			pred_t_raw = pred_t_raw.mean(dim=1, keepdim=False)

		# Reshape model outputs for later loss, metrics calculations:
		if (self.model_type == 'clf'):
			if (self.t_params['loss'] in ('clf-dnll',)):
				pred_t_loss = out_dist
				pred_t = pred_t_raw.detach().clone().clamp(0.0, 1.0)
				pred_t_ret = (pred_t - .5) * 2
			elif (self.t_params['loss'] in ('clf-ce',)):
				pred_t_loss = pred_t_raw
				pred_t_smax = F.softmax(pred_t_raw.detach().clone(), dim=-1)
				pred_t_conf, pred_t = pred_t_smax.max(dim=-1, keepdim=False)
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

		if (is_valid(prior_dist) and is_valid(post_dist)):
			# [C, S] -> [C, n]
			kldiv = torch.distributions.kl_divergence(post_dist, prior_dist) \
				.sum(dim=-1, keepdim=True).repeat(1, pred_t_raw.shape[0])
			kldiv /= pred_t_raw.shape[0]
		else:
			kldiv = 0

		try:
			aim_t_loss = aim_t
			# if (is_valid(prec := self.t_params['label_precision'])):
			# 	ftype = {
			# 		16: torch.float16,
			# 		32: torch.float32,
			# 		64: torch.float64
			# 	}.get(prec, 16)
			# 	aim_t_loss = aim_t.clone().to(ftype)
			model_loss = self.loss(pred_t_loss, aim_t_loss)
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > loss()\n",
				sys.exc_info()[0], err)
			print(f'{self.loss=}')
			print(f'{pred_t_loss.shape=}')
			print(f'{aim_t.shape=}')
			print(f'{pred_t.shape=}')
			print(f'{pred_t_loss.shape=}')
			print(f'{pred_t_conf.shape=}')
			print(f'{pred_t_dir.shape=}')
			print(f'{pred_t_ret.shape=}')
			raise err

		# if (train_mode):
		# 	print(f'{torch.distributions.kl_divergence(post_dist, prior_dist).shape=}')
		# 	print(f'{model_loss.shape=}')
		# 	print(f'{model_loss=}')
		# 	sys.exit()
		# 	
		# kl = tf.reduce_sum(
		# tf.contrib.distributions.kl_divergence(posterior, prior), 
		# axis=-1, keepdims=True)
		# kl = tf.tile(kl, [1, num_targets])
		# loss = - tf.reduce_mean(log_p - kl / tf.cast(num_targets, tf.float32))

		try:
			np_loss = (model_loss + kldiv).mean()
			# if (train_mode): raise Exception()
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > loss()\n",
				sys.exc_info()[0], err)
			print(f'{self.loss=}')
			print(f'{model_loss.shape=}')
			if (not is_type(kldiv, int)):
				print(f'{kldiv.shape=}')
			else:
				print(f'{kldiv=}')
			print(f'{out_dist=}')
			print(f'{aim_t.shape=}')
			print(f'{pred_t.shape=}')
			if (self.t_params['loss'] not in ('clf-dnll', 'reg-dnll')):
				print(f'{pred_t_loss.shape=}')
			else:
				print(f'{pred_t_loss=}')
			print(f'{pred_t_conf.shape=}')
			print(f'{pred_t_dir.shape=}')
			print(f'{pred_t_ret.shape=}')
			raise err

		for met in self.epoch_metrics[epoch_type].values():
			try:
				met.update(pred_t.cpu(), aim_t.cpu())
			except Exception as err:
				print("Error! pl_np.py > NPModel > forward_step() > met.update()\n",
					sys.exc_info()[0], err)
				print(f'{met=}')
				print(f'{pred_t.shape=}')
				print(f'{aim_t.shape=}')
				print(f'{pred_t.shape=}')
				print(f'{pred_t_loss.shape=}')
				if (self.t_params['loss'] not in ('clf-dnll', 'reg-dnll')):
					print('pred_t_loss.shape:', pred_t_loss.shape)
				print(f'{pred_t_conf.shape=}')
				print(f'{pred_t_dir.shape=}')
				print(f'{pred_t_ret.shape=}')
				print(f'{model_loss.shape=}')
				print(f'{kldiv.shape=}')
				print(f'{np_loss.shape=}')
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
					print(f'{pred_t.shape=}')
					print(f'{pred_t_loss.shape=}')
					print(f'{pred_t_conf.shape=}')
					print(f'{pred_t_dir.shape=}')
					print(f'{pred_t_ret.shape=}')
					print(f'{model_loss.shape=}')
					print(f'{kldiv.shape=}')
					print(f'{np_loss.shape=}')
					raise err

		return {'loss': np_loss}

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
				'train_target_overlap': None,
				'train_sample_context_size': True,
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
				'train_target_overlap': None,
				'train_sample_context_size': True,
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

