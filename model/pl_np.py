import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from common_util import is_type, is_valid, pt_resample_values, get_fn_params
from model.common import PYTORCH_LOSS_MAPPING
from model.pl_generic import GenericModel


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
	def __init__(self, pt_model_fn, params_m, params_d, fshape, splits=('train', 'val')):
		"""
		Init method

		Args:
			pt_model_fn (function): neural process pytorch model callback
			params_m (dict): dictionary of model hyperparameters
			params_d (dict): dictionary of data hyperparameters
			fshape (tuple): the shape of a single feature observation,
				this is usually the model input shape
			splits (tuple): which splits to init metric objects for
		"""
		super().__init__(pt_model_fn, params_m, params_d, fshape, splits=splits)

	def __init_model__(self, pt_model_fn, fshape):
		"""
		Args:
			pt_model_fn (torch.nn.Module): pytorch model constructor
			fshape (tuple): shape of a single feature observation
		"""
		model_params = get_fn_params(pt_model_fn, self.params_m)
		self.model = pt_model_fn(
			in_shape=fshape,
			context_size=self.params_d['context_size'],
			target_size=self.params_d['target_size'],
			**model_params
		)
		self.precision = 32

	def forward(self, batch):
		"""
		Run input through model and return output.
		Use at test time only.
		"""
		ic, xc, yc, zc, it, xt, yt, zt = batch
		prior, post, pred = self.model(xc, yc, xt, target_y=None)
		return (ic, yc), (it, yt), (prior, post, pred)

	def forward_eval(self, dl):
		self.eval()
		with torch.no_grad():
			outs = [self.forward(b) for b in dl]

		ic = torch.cat([i[0][0].flatten() for i in outs])
		yc = torch.cat([i[0][1].flatten() for i in outs])
		it = torch.cat([i[1][0].flatten() for i in outs])
		yt = torch.cat([i[1][1].flatten() for i in outs])
		prior = [i[2][0] for i in outs]
		post = [i[2][1] for i in outs]
		pred = [i[2][2] for i in outs]
		return (ic, yc), (it, yt), (prior, post, pred)

	def pred_df(self, dl, index):
		outs = self.forward_eval(dl)
		ic, yc = outs[0]
		it, yt = outs[1]
		prior, post, out = outs[2]
		# print(f'{it=}')
		# print(f'{it.sort()[0]=}')
		# print(f'{it.sort()[1]=}')
		# print(torch.equal(it, it.sort()[0]))
		# sys.exit()
		pred_mean = torch.cat([i.mean for i in out]).flatten()
		pred_std = torch.cat([i.variance for i in out]).flatten().sqrt()
		assert all(len(x)==len(ic) for x in (yc, it, yt, pred_mean, pred_std))
		return pd.DataFrame.from_dict({
			"it": index[it], "yt": yt, "ic": index[ic], "yc": yc,
			"pred_mean": pred_mean, "pred_std": pred_std
		}).drop_duplicates("it").set_index("it").sort_index()

	def forward_step(self, batch, batch_idx, epoch_type):
		"""
		Run forward pass, calculate step loss, and calculate step metrics.
		"""
		train_mode = epoch_type == 'train'
		ic, xc, yc, zc, it, xt, yt, zt = batch
		# print(f'{xc.shape=} {xt.shape=}')
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
			pred_t, pred_t_loss = self.prepare_pred(out_dist, train_mode)
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

		return {'loss': np_loss, 'kldiv': kldiv.mean()}

	def prepare_pred(self, out_dist, train_mode):
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

