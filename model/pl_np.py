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

from common_util import is_type, is_valid, pt_resample_values
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
		train_shuffle (bool): whether or not to shuffle the order of the training batches
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
		if (is_valid(cs:=self.params_t['context_size']) and is_valid(ts:=self.params_t['target_size'])):
			assert cs+ts == self.params_t['batch_size'], \
				"context and target sizes must add up to batch_size, \
				use train_target_overlap to add a training overlap"

	def get_context_target(self, batch, train_mode=False):
		"""
		Return context and target points.
		If in regressor mode, y and z are identical.
		"""
		x, y, z = batch
		ctx = self.params_t['context_size'] or self.params_t['batch_size']//2
		tgt = self.params_t['target_size'] or self.params_t['batch_size'] - ctx

		if (train_mode):
			if (self.params_t['train_sample_context_size']):
				ctx = np.random.randint(low=1, high=self.params_t['batch_size'], \
					size=None, dtype=int)
				tgt = self.params_t['batch_size'] - ctx

			tto = self.params_t['train_target_overlap']
			tend = None
			if (is_type(tto := self.params_t['train_target_overlap'], int) and tto!=0):
				tgt += tto
				if (tto > 0):
					tend = -tto

			if (is_valid(ts := self.params_t['train_resample']) and \
				self.model_type == 'clf'):
				ctx_idx = pt_resample_values(y[:ctx], n=ts, shuffle=True) \
					.to(y.device)

				# XXX Disable train target resampling:
				tgt_idx = pt_resample_values(y[-tgt:tend], n=ts, shuffle=True) \
					.to(y.device)

				return x[:ctx].index_select(dim=0, index=ctx_idx), \
					y[:ctx].index_select(dim=0, index=ctx_idx), \
					z[:ctx].index_select(dim=0, index=ctx_idx), \
					x[-tgt:tend].index_select(dim=0, index=tgt_idx), \
					y[-tgt:tend].index_select(dim=0, index=tgt_idx), \
					z[-tgt:tend].index_select(dim=0, index=tgt_idx)
			else:
				return x[:ctx], y[:ctx], z[:ctx], \
					x[-tgt:tend], y[-tgt:tend], z[-tgt:tend]
		else:
			return x[:ctx], y[:ctx], z[:ctx], x[-tgt:], y[-tgt:], z[-tgt:]

	def forward(self, context_x, context_a, target_x, target_a=None, sample_out=False):
		"""
		Run input through model and return output.
		Use at test time only.
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
		dist_type = self.params_m['decoder_params']['dist_type']

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

		if (self.params_t['sample_out'] and train_mode and out_dist.has_rsample and not (self.params_m['loss'] in ('clf-dnll',))):
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
		elif (self.model_type == 'reg'):
			if (self.params_m['loss'] in ('reg-dnll',)):
				pred_t_loss = out_dist
				pred_t = pred_t_raw.detach().clone()
			elif (self.params_m['loss'] in ('reg-mae', 'reg-mse')):
				pred_t_loss = pred_t_raw
				pred_t = pred_t_raw.detach().clone()
			elif (self.params_m['loss'] in ('reg-sharpe',)):
				pred_t_loss = pred_t_raw
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

		if (is_valid(prior_dist) and is_valid(post_dist)):
			# [C, S] -> [C, n]
			kldiv = torch.distributions.kl_divergence(post_dist, prior_dist) \
				.sum(dim=-1, keepdim=True).repeat(1, pred_t_raw.shape[0])
			kldiv /= pred_t_raw.shape[0]
			kldiv = kldiv * self.params_t['kl_weight']
		else:
			kldiv = torch.zeros(1, device=pred_t_raw.device)

		try:
			aim_t_loss = aim_t
			if (self.params_m['loss'] in ('clf-bce', 'clf-bcel') or (self.params_m['loss'] == 'clf-dnll' and dist_type in ('bernoulli', 'cbernoulli'))):
				aim_t_loss = aim_t_loss.float()
			# aim_t_loss = aim_t.type_as(pred_t_loss)
			# aim_t_loss = aim_t.to(torch.)
			# if (is_valid(prec := self.get_precision())):
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
			print(f'{pred_t_loss.shape=}, {pred_t_loss.dtype=}')
			print(f'{aim_t_loss.shape=}, {aim_t_loss.dtype=}')
			print(f'{pred_t.shape=}')
			print(f'{pred_t_loss.shape=}')
			print(f'{pred_t_conf.shape=}')
			print(f'{pred_t_dir.shape=}')
			print(f'{pred_t_bet.shape=}')
			raise err

		# if (train_mode):
		# 	print(f'{torch.distributions.kl_divergence(post_dist, prior_dist).shape=}')
		# 	print(f'{model_loss.shape=}')
		# 	print(f'{model_loss=}')
		# 	sys.exit()

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
			if (self.params_m['loss'] not in ('clf-dnll', 'reg-dnll')):
				print(f'{pred_t_loss.shape=}')
			else:
				print(f'{pred_t_loss=}')
			print(f'{pred_t_conf.shape=}')
			print(f'{pred_t_dir.shape=}')
			print(f'{pred_t_bet.shape=}')
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
				if (self.params_m['loss'] not in ('clf-dnll', 'reg-dnll')):
					print('pred_t_loss.shape:', pred_t_loss.shape)
				print(f'{pred_t_conf.shape=}')
				print(f'{pred_t_dir.shape=}')
				print(f'{pred_t_bet.shape=}')
				print(f'{model_loss.shape=}')
				print(f'{kldiv.shape=}')
				print(f'{np_loss.shape=}')
				raise err

		if (is_valid(self.epoch_returns)):
			for ret in self.epoch_returns[epoch_type].values():
				try:
					ret.update(pred_t_bet.cpu(), zt.cpu())
				except Exception as err:
					print("Error! pl_np.py > NPModel > forward_step() > ret.update()\n",
						sys.exc_info()[0], err)
					print(f'{ret=}')
					print(f'{pred_t_bet.shape=}')
					print(f'{zt.shape=}')
					print(f'{pred_t.shape=}')
					print(f'{pred_t_loss.shape=}')
					print(f'{pred_t_conf.shape=}')
					print(f'{pred_t_dir.shape=}')
					print(f'{pred_t_bet.shape=}')
					print(f'{model_loss.shape=}')
					print(f'{kldiv.shape=}')
					print(f'{np_loss.shape=}')
					raise err

		return {'loss': np_loss, 'kldiv': kldiv.mean()}

	def aggregate_log_epoch_loss(self, outputs, epoch_type):
		"""
		Aggregate step losses / kldivs and log them.
		"""
		step_losses = [d['loss'] and d['loss'].cpu() for d in outputs]
		step_kldivs = [d['kldiv'] and d['kldiv'].cpu() for d in outputs]
		epoch_loss = None
		epoch_kldiv = None
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
		if (self.params_t['sample_out'] and self.params_m['decoder_params']['dist_type'] == 'beta'):
			precision = 32
		elif (self.params_m['use_lat_path'] and \
			(self.params_m['sample_latent_post'] or self.params_m['sample_latent_prior']) \
			and self.params_m['lat_encoder_params']['dist_type'] == 'beta'):
			precision = 32
		return precision

