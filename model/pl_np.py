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

from common_util import is_type, is_valid, pt_resample_values, get_fn_params
from model.common import PYTORCH_LOSS_MAPPING
from model.pl_generic import GenericModel
from model.metrics_util import SimulatedReturn


class NPModel(GenericModel):
	"""
	Neural Process Pytorch Lightning Wrapper.

	Training Hyperparameters:
		window_size (int): number of observations in the last dimension of the input tensor
		epochs (int): number training epochs
		batch_size (int): batch (or batch window) size
		batch_step_size (int): batch window step size.
		context_size
		target_size
		overlap_size
		shuffle (bool): whether or not to shuffle the order of the training batches
		resample_context (bool): whether or not to resample the context set without replacement (adds duplicates / bias during training)
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
	def __init__(self, pt_model_fn, params_m, params_t, fshape, splits=('train', 'val')):
		"""
		Init method

		Args:
			pt_model_fn (function): neural process pytorch model callback
			params_m (dict): dictionary of model hyperparameters
			params_t (dict): dictionary of training hyperparameters
			fshape (tuple): the shape of a single feature observation,
				this is usually the model input shape
			splits (tuple): which splits to init metric objects for
		"""
		super().__init__(pt_model_fn, params_m, params_t, fshape, splits=splits)

	def __init_model__(self, pt_model_fn, fshape):
		"""
		Args:
			pt_model_fn (torch.nn.Module): pytorch model constructor
			fshape (tuple): shape of a single feature observation
		"""
		model_params = get_fn_params(pt_model_fn, self.params_m)
		self.model = pt_model_fn(
			in_shape=fshape,
			context_size=self.params_t['context_size'],
			target_size=self.params_t['target_size'],
			**model_params
		)

	def forward(self, batch):
		"""
		Run input through model and return output.
		Use at test time only.
		"""
		ic, xc, yc, zc, it, xt, yt, zt = batch
		prior_dist, post_dist, out_dist = self.model(xc, yc, xt, target_y=yt)
		return out_dist.mean.squeeze(), out_dist.stddev.squeeze()

	def forward_step(self, batch, batch_idx, epoch_type):
		"""
		Run forward pass, calculate step loss, and calculate step metrics.
		"""
		train_mode = epoch_type == 'train'
		ic, xc, yc, zc, it, xt, yt, zt = batch
		# print(f'{xc.shape[0]=} {xt.shape[0]=}')
		dist_type = self.params_m['decoder_params']['dist_type']

		try:
			prior_dist, post_dist, out_dist = self.model(xc, yc, xt, \
				target_y=yt if (train_mode) else None)
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > model()\n",
				sys.exc_info()[0], err)
			print(f'{train_mode=}')
			print(f'{xc.shape=},{yc.shape=},{zc.shape=}')
			raise err

		try:
			pred_t, pred_t_loss = self.prepare_pred(out_dist)
			model_loss = self.loss(pred_t_loss, yt)
			if (model_loss.ndim > 1):
				model_loss = model_loss.mean(1)
			kldiv = torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1)\
				if (prior_dist and post_dist) else torch.zeros_like(model_loss)
			np_loss = (model_loss + kldiv * self.params_m['kl_beta']).mean()
		except Exception as err:
			print("Error! pl_np.py > NPModel > forward_step() > loss()\n",
				sys.exc_info()[0], err)
			print(f'{self.loss=}')
			# print(f'{pred_t_loss.shape=}, {pred_t_loss.dtype=}')
			print(f'{yt.shape=}, {yt.dtype=}')
			raise err

		for met in self.epoch_metrics[epoch_type].values():
			met.update(pred_t.cpu(), yt.cpu())

		if (is_valid(self.epoch_returns)):
			for ret in self.epoch_returns[epoch_type].values():
				ret.update(pred_t_bet.cpu(), zt.cpu())

		return {'loss': np_loss, 'kldiv': kldiv.mean()}

	def prepare_pred(self, out_dist):
		"""
		XXX
		Reshape model outputs for later loss, metrics calculations
		"""
		if (self.params_m['sample_out'] and train_mode and out_dist.has_rsample \
			and not (self.params_m['loss'] in ('clf-dnll',))):
			pred_t = out_dist.rsample().squeeze()
		else:
			pred_t = out_dist.mean.squeeze()

		if (self.model_type == 'clf'):
			pred_t_loss = self.prepare_pred_clf(out_dist, pred_t)
		elif (self.model_type == 'reg'):
			pred_t_loss = self.prepare_pred_reg(out_dist, pred_t)

		return pred_t, pred_t_loss

	def prepare_pred_reg(self, out_dist, pred_t):
		"""
		XXX
		"""
		if (self.params_m['loss'] in ('reg-dnll',)):
			pred_t_loss = out_dist
		elif (self.params_m['loss'] in ('reg-mae', 'reg-mse')):
			pred_t_loss = pred_t
		elif (self.params_m['loss'] in ('reg-sharpe',)):
			pred_t_loss = pred_t
			if (pred_t_loss.ndim == 1):
				preds = ((1-pred_t_loss).unsqueeze(-1), pred_t_loss.unsqueeze(-1))
				pred_t_loss = torch.hstack(preds)
			elif (pred_t_loss.ndim < self.params_m['num_classes']):
				# sum the probs and take complement
				raise NotImplementedError()
			pred_t_smax = F.softmax(pred_t_loss, dim=-1)
			pred_t_conf, pred_t = pred_t_smax.max(dim=-1, keepdim=False)
			pred_t_dir = pred_t.clone()
			pred_t_dir[pred_t_dir==0] = -1
			pred_t_bet = pred_t_dir * pred_t_conf
			pred_t_loss = pred_t_bet
		return pred_t_loss

	def prepare_pred_clf(self, out_dist, pred_t_raw):
		"""
		XXX
		"""
		raise NotImplementedError()
		if (self.params_m['loss'] in ('clf-dnll',)):
			pred_t_loss = out_dist
			pred_t = pred_t_raw.detach().clone()
			if (dist_type == 'beta'):
				pred_t = torch.sigmoid(pred_t)
			else:
				pred_t = pred_t.clamp(0.0, 1.0)
			pred_t_bet = (pred_t - .5) * 2
		elif (self.params_m['loss'] in ('clf-bcel',)):
			pred_t = pred_t_raw.detach().clone()[:, 1]
			pred_t_loss = pred_t
			pred_t_bet = (pred_t - .5) * 2
			pred_t_conf = pred_t_bet.abs()
			pred_t_dir = pred_t_bet.sign()
		elif (self.params_m['loss'] in ('clf-bce',)):
			pred_t = pred_t_raw.detach().clone()[:, 1]
			pred_t_loss = pred_t
			pred_t_bet = (pred_t - .5) * 2
			pred_t_conf = pred_t_bet.abs()
			pred_t_dir = pred_t_bet.sign()
		elif (self.params_m['loss'] in ('clf-ce',)):
			pred_t_loss = pred_t_raw
			if (pred_t_loss.ndim == 1):
				preds = ((1-pred_t_loss).unsqueeze(-1), pred_t_loss.unsqueeze(-1))
				pred_t_loss = torch.hstack(preds)
			elif (pred_t_loss.ndim < self.params_m['num_classes']):
				# sum the probs and take complement
				raise NotImplementedError()
			pred_t_smax = F.softmax(pred_t_loss.detach().clone(), dim=-1)
			pred_t_conf, pred_t = pred_t_smax.max(dim=-1, keepdim=False)
			pred_t_dir = pred_t.detach().clone()
			pred_t_dir[pred_t_dir==0] = -1
			pred_t_bet = pred_t_dir * pred_t_conf
		elif (self.params_m['loss'] in ('clf-nll',)):
			pred_t_loss = pred_t_raw
			if (pred_t_loss.ndim == 1):
				preds = ((1-pred_t_loss).unsqueeze(-1), pred_t_loss.unsqueeze(-1))
				pred_t_loss = torch.hstack(preds)
				pred_t_prob = pred_t_loss.detach().clone()
				pred_t_loss = pred_t_loss.log()
			elif (pred_t_loss.ndim < self.params_m['num_classes']):
				# sum the probs and take complement
				raise NotImplementedError()
			else:
				raise ValueError()
			pred_t_conf, pred_t = pred_t_prob.max(dim=-1, keepdim=False)
			pred_t_dir = pred_t.detach().clone()
			pred_t_dir[pred_t_dir==0] = -1
			pred_t_bet = pred_t_dir * pred_t_conf
		else:
			pred_t_loss = pred_t_raw
			raise NotImplementedError()
		return pred_t_loss

	def aggregate_log_epoch_loss(self, outputs, epoch_type):
		"""
		Aggregate step losses / kldivs and log them.
		"""
		step_losses = [d['loss'].cpu() for d in outputs]
		step_kldivs = [d['kldiv'].cpu() for d in outputs]
		epoch_loss, epoch_kldiv = None, None
		if (all(step_losses)):
			epoch_loss = torch.mean(torch.stack(step_losses), dim=0)
			epoch_kldiv = torch.mean(torch.stack(step_kldivs), dim=0)

			self.log('epoch', self.trainer.current_epoch, prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			self.log(f'{epoch_type}_loss', epoch_loss, prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			self.log(f'{epoch_type}_kldiv', epoch_kldiv, prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)

	def get_precision(self):
		"""
		Workaround for pytorch Dirichlet/Beta not supporting half precision.
		"""
		precision = 16
		if (self.params_m['sample_out'] and self.params_m['decoder_params']['dist_type'] == 'beta'):
			precision = 32
		elif (self.params_m['use_lat_path'] and \
			(self.params_m['sample_latent_post'] or self.params_m['sample_latent_prior']) \
			and self.params_m['lat_encoder_params']['dist_type'] == 'beta'):
			precision = 32
		return precision

