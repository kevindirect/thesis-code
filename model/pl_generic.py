import sys
import os
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl # PL ver 1.2.3
import torchmetrics as tm

from common_util import load_df, dump_df, is_valid, isnt, rectify_json, dump_json, get_fn_params
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_LOSS_MAPPING, PYTORCH_OPT_MAPPING, PYTORCH_SCH_MAPPING


class GenericModel(pl.LightningModule):
	"""
	Generic Pytorch Lightning Wrapper.
	"""
	def __init__(self, pt_model_fn, params_m, params_d, fshape, splits=('train', 'val')):
		"""
		Init method

		Args:
			pt_model_fn (function): pytorch model callback
			params_m (dict): dictionary of model hyperparameters
			params_d (dict): dictionary of data hyperparameters
			fshape (tuple): the shape of a single feature observation,
				this is usually the model input shape
			splits (tuple): which splits to init metric objects for
		"""
		super().__init__()
		self.name = f'{self._get_name()}_{pt_model_fn.__name__}'
		self.params_m, self.params_d = params_m, params_d
		self.model_type = self.params_m['loss'].split('-')[0]
		self.__init_loss_fn__()
		self.__init_model__(pt_model_fn, fshape)
		self.__init_metrics__(splits)
		self.__init_trackers__(splits)
		#self.example_input_array = torch.rand(10, *fshape, dtype=torch.float32) * 100

	def __init_loss_fn__(self, reduction='none'):
		if (is_valid(loss_fn := PYTORCH_LOSS_MAPPING.get(self.params_m['loss'], None))):
			if (self.model_type=='clf' and is_valid(cw := self.params_m['class_weights'])):
				self.loss = loss_fn(reduction=reduction, weight=torch.tensor(cw))
			else:
				self.loss = loss_fn(reduction=reduction)
		else:
			logging.info('no loss function set in pytorch lightning')

	def __init_model__(self, pt_model_fn, fshape):
		"""
		Args:
			pt_model_fn (torch.nn.Module): pytorch model constructor
			fshape (tuple): the shape of a single feature observation,
				this is usually the model input shape
		"""
		model_params = get_fn_params(pt_model_fn, self.params_m)
		self.model = pt_model_fn(in_shape=fshape, **model_params)
		self.precision = 32

	def __init_metrics__(self, splits):
		"""
		'micro' weights by class frequency, 'macro' weights classes equally
		"""
		if (self.model_type == 'clf'):
			if (self.params_m['loss'] in ('clf-ce', 'clf-nll')):
				num_classes = self.params_m['num_classes'] or self.params_m['out_size'] + 1
			else:
				num_classes = self.params_m['num_classes'] or self.params_m['out_size']
			self.epoch_metrics = {
				epoch_type: {
					f'{self.model_type}_accuracy': tm.Accuracy(compute_on_step=False),
					f'{self.model_type}_precision': tm.Precision(num_classes=num_classes,
						average='macro', compute_on_step=False),
					f'{self.model_type}_recall': tm.Recall(num_classes=num_classes,
						average='macro', compute_on_step=False),
					f'{self.model_type}_f1': tm.F1Score(num_classes=num_classes,
						average='macro', compute_on_step=False),
					# f'{self.model_type}_f0.5': tm.FBetaScore(num_classes=num_classes, beta=0.5,
					# 	average='micro', compute_on_step=False),
				}
				for epoch_type in splits
			}
		elif (self.model_type == 'reg'):
			# if (self.params_m['loss'] in ('reg-sharpe',)):
			# 	num_classes = self.params_m['num_classes'] or self.params_m['out_size'] + 1
			# else:
			# 	assert('num_classes' not in self.params_m or isnt(self.params_m['num_classes']))
			self.epoch_metrics = {
				epoch_type: {
					f'{self.model_type}_mae': tm.MeanAbsoluteError(compute_on_step=False),
					f'{self.model_type}_mse': tm.MeanSquaredError(compute_on_step=False),
				}
				for epoch_type in splits
			}

	def __init_trackers__(self, splits):
		self.epoch_trackers = None

	def configure_optimizers(self):
		"""
		Construct and return optimizers
		"""
		opt_fn = PYTORCH_OPT_MAPPING.get(self.params_m['opt']['name'])
		opt = opt_fn(self.parameters(), **self.params_m['opt']['kwargs'])
		return opt
		#sch_fn = PYTORCH_SCH_MAPPING.get(self.params_m['sch']['name'])
		#sch = sch_fn(opt, **self.params_m['sch']['kwargs'])
		#return [opt], [sch]

	def forward(self, x):
		"""
		Run input through model and return output. Used at inference time only.

		Use pl.Trainer.predict to get predictions based on input data.
		Use pl.Trainer.{validate, test} to evalute the model over validation/test sets.
		"""
		try:
			return self.model(x)
		except Exception as err:
			print("Error! pl_generic.py > GenericModel > forward() > model()\n",
				sys.exc_info()[0], err)
			print(f'{x.shape=}')
			print(f'{y.shape=}')
			print(f'{z.shape=}')
			raise err

	def forward_eval(self, dl):
		self.eval()
		with torch.no_grad():
			outs = [self(b) for b in dl]
		return outs

	def forward_step(self, batch, batch_idx, epoch_type):
		"""
		Run forward pass, calculate step loss, and calculate step metrics. Used for training.
		"""
		x, y, z = batch
		try:
			pred_raw = self.model(x)
		except Exception as err:
			print("Error! pl_generic.py > GenericModel > forward_step() > model()\n",
				sys.exc_info()[0], err)
			print(f'{x.shape=}')
			print(f'{y.shape=}')
			print(f'{z.shape=}')
			raise err

		# Reshape model outputs for later loss, metrics calculations:
		if (self.model_type == 'clf'):
			actual = y
			if (self.params_m['loss'] in ('clf-bce',)):
				pred_t_loss = pred_t_raw
				pred_t_loss = F.sigmoid(pred_t_loss, dim=-1)
				pred_t = pred_t_loss.detach().clone()
				pred_t_ret = (pred_t - .5) * 2
			elif (self.params_m['loss'] in ('clf-ce',)):
				pred_t_loss = pred_t_raw
				if (pred_t_loss.ndim == 1):
					preds = (pred_t_loss.unsqueeze(-1), (1-pred_t_loss).unsqueeze(-1))
					pred_t_loss = torch.hstack(preds)
				elif (pred_t_loss.ndim < self.params_m['num_classes']):
					# sum the probs and take complement
					raise NotImplementedError()
				pred_t_smax = F.softmax(pred_t_loss.detach().clone(), dim=-1)
				pred_t_conf, pred_t = pred_t_smax.max(dim=-1, keepdim=False)
				pred_t_dir = pred_t.detach().clone()
				pred_t_dir[pred_t_dir==0] = -1
				pred_t_ret = pred_t_dir * pred_t_conf
			else:
				raise NotImplementedError()
		elif (self.model_type == 'reg'):
			actual = z
			pred_loss = pred_raw.squeeze()
			pred_ret = pred = pred_loss.detach().clone()

		try:
			model_loss = self.loss(pred_loss, actual)
		except Exception as err:
			print("Error! pl_generic.py > GenericModel > forward_step() > loss()\n",
				sys.exc_info()[0], err)
			print(f'{self.loss=}')
			print(f'{pred_loss.shape=}')
			print(f'{actual.shape=}')
			raise err

		for met in self.epoch_metrics[epoch_type].values():
			try:
				met.update(pred.cpu(), actual.cpu())
			except Exception as err:
				print("Error! pl_generic.py > GenericModel > forward_step() > met.update()\n",
					sys.exc_info()[0], err)
				print(f'{met=}')
				print(f'{pred.shape=}')
				print(f'{actual.shape=}')
				raise err

		return {'loss': model_loss}

	def aggregate_log_epoch_loss(self, outputs, epoch_type):
		"""
		Aggregate step losses and log them.
		"""
		step_losses = [d['loss'] and d['loss'].cpu() for d in outputs]
		epoch_loss = None
		if (all(step_losses)):
			epoch_loss = torch.mean(torch.stack(step_losses), dim=0)

			self.log('epoch', self.trainer.current_epoch, prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			self.log(f'{epoch_type}_loss', epoch_loss, prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)

	def compute_log_epoch_metrics(self, epoch_type):
		"""
		Compute and log the running metrics.
		"""
		for name, met in self.epoch_metrics[epoch_type].items():
			self.log(f'{epoch_type}_{name}', met.compute(), prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)

	def reset_metrics(self, epoch_type):
		"""
		Reset/Clear the running metrics.
		"""
		for name, met in self.epoch_metrics[epoch_type].items():
			met.reset()

		if (is_valid(self.epoch_trackers)):
			for split, tracker in self.epoch_trackers.items():
				tracker.reset()

	def on_train_epoch_start(self, epoch_type='train'):
		"""
		Clear training metrics for new epoch.
		"""
		self.reset_metrics(epoch_type)

	def training_step(self, batch, batch_idx, epoch_type='train'):
		"""
		Compute and return training step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def training_epoch_end(self, outputs, epoch_type='train'):
		"""
		Aggregate training step losses and metrics and log them all.
		"""
		self.aggregate_log_epoch_loss(outputs, epoch_type)
		self.compute_log_epoch_metrics(epoch_type)

	def on_validation_epoch_start(self, epoch_type='val'):
		"""
		Clear validation metrics for new epoch.
		"""
		self.reset_metrics(epoch_type)

	def validation_step(self, batch, batch_idx, epoch_type='val'):
		"""
		Compute and return validation step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def validation_epoch_end(self, outputs, epoch_type='val'):
		"""
		Aggregate validation step losses and metrics and log them all.
		"""
		self.aggregate_log_epoch_loss(outputs, epoch_type)
		self.compute_log_epoch_metrics(epoch_type)

	def on_test_epoch_start(self, epoch_type='test'):
		"""
		Clear test metrics for new epoch.
		"""
		self.reset_metrics(epoch_type)

	def test_step(self, batch, batch_idx, epoch_type='test'):
		"""
		Compute and return test step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def test_epoch_end(self, outputs, epoch_type='test'):
		"""
		Aggregate test step losses and metrics and log_epoch them all.
		"""
		self.aggregate_log_epoch_loss(outputs, epoch_type)
		self.compute_log_epoch_metrics(epoch_type)

