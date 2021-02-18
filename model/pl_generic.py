"""
Kevin Patel
"""
import sys
import os
import logging
from inspect import getfullargspec

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl # PL ver 1.0.4
from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall, Fbeta

from common_util import load_df, dump_df, is_valid, isnt
from model.common import WIN_SIZE, PYTORCH_ACT_MAPPING, PYTORCH_LOSS_MAPPING, PYTORCH_OPT_MAPPING, PYTORCH_SCH_MAPPING
from model.metrics_util import SimulatedReturn
from model.model_util import OutputBlock


class GenericModel(pl.LightningModule):
	"""
	Generic Pytorch Lightning Wrapper.

	Training Hyperparameters:
		window_size (int): number of observations in the last dimension of the input tensor
		feat_dim (int): dimension of resulting feature tensor, if 'None' doesn't reshape
		epochs (int): max number of training epochs
		batch_size (int): batch (or batch window) size
		batch_step_size (int): batch window step size.
			if this is None DataLoader uses its own default sampler,
			otherwise WindowBatchSampler is used as batch_sampler
		train_shuffle (bool): whether or not to shuffle the order of the training batches
		loss (str): name of the loss function to use,
			'clf' (classsifer) and 'reg' (regressor) are 'dummy losses' that only affect how
			the labels/targets are preprocessed (look at model.train_util.py)
		class_weights (list): loss function class weights of size C (optional)
		opt (dict): pytorch optimizer settings
			name (str): name of optimizer to use
			kwargs (dict): any keyword arguments to the optimizer constructor
		sch (dict): pytorch scheduler settings
			name (str): name of scheduler to use
			kwargs (dict): any keyword arguments to the scheduler constructor
		num_workers (int>=0): DataLoader option - number cpu workers to attach
		pin_memory (bool): DataLoader option - whether to pin memory to gpu
	"""
	def __init__(self, pt_model_fn, m_params, t_params, fobs,
		epoch_metric_types=('train', 'val')):
		"""
		Init method

		Args:
			pt_model_fn (function): pytorch model callback
			m_params (dict): dictionary of model hyperparameters
			t_params (dict): dictionary of training hyperparameters
			fobs (tuple): the shape of a single feature observation,
				this is usually the model input shape
			epoch_metric_types (tuple): which epoch types to init metric objects for
		"""
		super().__init__()
		self.name = f'{self._get_name()}_{pt_model_fn.__name__}'
		self.m_params, self.t_params = m_params, t_params
		self.__init_loss_fn__()
		self.__init_model__(pt_model_fn, fobs)
		self.__init_loggers__(epoch_metric_types)
		#self.example_input_array = torch.rand(10, *fobs, dtype=torch.float32) * 100

	def __init_loss_fn__(self):
		if (is_valid(loss_fn := PYTORCH_LOSS_MAPPING.get(self.t_params['loss'], None))):
			if (is_valid(self.t_params['class_weights'])):
				self.loss = loss_fn(weight=self.t_params['class_weights'])
			else:
				self.loss = loss_fn()
		else:
			logging.info('no loss function set in pytorch lightning')

	def __init_model__(self, pt_model_fn, fobs):
		"""
		Args:
			pt_model_fn (torch.nn.Module): pytorch model constructor
			fobs (tuple): the shape of a single feature observation,
				this is usually the model input shape
		"""
		model_params = {k: v for k, v in self.m_params.items() \
			if (k in getfullargspec(pt_model_fn).args)}

		self.model = pt_model_fn(in_shape=fobs, **model_params)
		if (is_valid(self.m_params.get('ob_out_shapes', None))):
			self.model = OutputBlock.wrap(self.model)

	def __init_loggers__(self, epoch_metric_types):
		# 'micro' weights by class frequency, 'macro' weights classes equally
		self.epoch_metrics = {
			epoch_type: {
				'accuracy': Accuracy(compute_on_step=False),
				'precision': Precision(num_classes=self.m_params['label_size'],
					average='macro', compute_on_step=False),
				'recall': Recall(num_classes=self.m_params['label_size'],
					average='macro', compute_on_step=False),
				# 'f0.5': Fbeta(num_classes=self.m_params['label_size'], beta=0.5,
				# 	average='micro', compute_on_step=False),
				'f1.0': Fbeta(num_classes=self.m_params['label_size'], beta=1.0,
					average='micro', compute_on_step=False),
				# 'f2.0': Fbeta(num_classes=self.m_params['label_size'], beta=2.0,
				# 	average='micro', compute_on_step=False),
			}
			for epoch_type in epoch_metric_types
		}
		self.epoch_returns = {
			epoch_type: {
				'br': SimulatedReturn(use_conf=False, compounded=False),
				'brc': SimulatedReturn(use_conf=False, compounded=True),
				'cr': SimulatedReturn(use_conf=True, use_kelly=False, compounded=False),
				'crc': SimulatedReturn(use_conf=True, use_kelly=False, compounded=True),
				'kr': SimulatedReturn(use_conf=True, use_kelly=True, compounded=False),
				'krc': SimulatedReturn(use_conf=True, use_kelly=True, compounded=True),
			}
			for epoch_type in epoch_metric_types
		}

	def configure_optimizers(self):
		"""
		Construct and return optimizers
		"""
		opt_fn = PYTORCH_OPT_MAPPING.get(self.t_params['opt']['name'])
		opt = opt_fn(self.parameters(), **self.t_params['opt']['kwargs'])
		return opt
		#sch_fn = PYTORCH_SCH_MAPPING.get(self.t_params['sch']['name'])
		#sch = sch_fn(opt, **self.t_params['sch']['kwargs'])
		#return [opt], [sch]

	def forward(self, x):
		"""
		Run input through the model.
		"""
		return self.model(x)

	def forward_step(self, batch, batch_idx, epoch_type='train'):
		"""
		Run forward pass, update step metrics, and return step loss.
		"""
		x, y, z = batch
		y_hat_raw = self.forward(x)
		if (self.t_params['loss'] in ('ce',)):
			y_hat = F.softmax(y_hat_raw, dim=-1)
		else:
			y_hat = y_hat_raw

		for met in self.epoch_metrics[epoch_type].values():
			met.update(y_hat.cpu().argmax(dim=-1), y.cpu())
		for ret in self.epoch_returns[epoch_type].values():
			ret.update(y_hat.cpu(), y.cpu(), z.cpu()) 	# Conf score

		return {'loss': self.loss(y_hat_raw, y)}

	def aggregate_loss_epoch_end(self, outputs, epoch_type='train'):
		"""
		Aggregate step losses into epoch loss and log it.
		"""
		step_losses = [d['loss'] and d['loss'].cpu() for d in outputs]
		epoch_loss = None
		if (all(step_losses)):
			epoch_loss = torch.mean(torch.stack(step_losses), dim=0)

			self.log('epoch', self.trainer.current_epoch, prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			self.log(f'{epoch_type}_loss', epoch_loss, prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)

	def compute_metrics_epoch_end(self, epoch_type='train'):
		"""
		Compute, log, and reset metrics at the end of an epoch.
		"""
		for name, met in self.epoch_metrics[epoch_type].items():
			self.log(f'{epoch_type}_{name}', met.compute(), prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			met.reset()
		for name, ret in self.epoch_returns[epoch_type].items():
			self.log_dict(ret.compute(pfx=epoch_type), prog_bar=False, \
				logger=True, on_step=False, on_epoch=True)
			ret.reset()

	def training_step(self, batch, batch_idx, epoch_type='train'):
		"""
		Compute and return training step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def training_epoch_end(self, outputs, epoch_type='train'):
		"""
		Aggregate training step losses and step metrics and log them all.
		"""
		self.aggregate_loss_epoch_end(outputs, epoch_type)
		self.compute_metrics_epoch_end(epoch_type)

	def validation_step(self, batch, batch_idx, epoch_type='val'):
		"""
		Compute and return validation step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def validation_epoch_end(self, outputs, epoch_type='val'):
		"""
		Aggregate validation step losses and step metrics and log them all.
		"""
		self.aggregate_loss_epoch_end(outputs, epoch_type)
		self.compute_metrics_epoch_end(epoch_type)

	def test_step(self, batch, batch_idx, epoch_type='test'):
		"""
		Compute and return test step loss.
		"""
		return self.forward_step(batch, batch_idx, epoch_type)

	def test_epoch_end(self, outputs, epoch_type='test'):
		"""
		Aggregate test step losses and step metrics and log them all.
		"""
		self.aggregate_loss_epoch_end(outputs, epoch_type)
		self.compute_metrics_epoch_end(epoch_type)

	@classmethod
	def fix_metrics_csv(cls, fname, dir_path):
		"""
		Fix Pytorch Lightning v10 logging rows in the same epoch on
		separate rows.
		"""
		csv_df = load_df(fname, dir_path=dir_path, data_format='csv')
		csv_df = csv_df.groupby('epoch').ffill().dropna(how='any')
		dump_df(csv_df, f'fix_{fname}', dir_path=dir_path, data_format='csv')
		logging.debug(f'fixed {fname}')

	@classmethod
	def suggest_params(cls, trial=None, num_classes=2):
		"""
		suggest training hyperparameters from an optuna trial object
		or return fixed default hyperparameters

		Pytorch recommends not using num_workers > 0 to return CUDA tensors
		because of the subtleties of CUDA multiprocessing, instead pin the
		memory to the GPU for fast data transfer:
		https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
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

			params = {
				'window_size': WIN_SIZE, #trial.suggest_int('window_size', 3, 30),
				'feat_dim': None,
				'train_shuffle': False,
				'epochs': trial.suggest_int('epochs', 200, 600, step=100),
				'batch_size': trial.suggest_int('batch_size', 128, 512, step=128),
				'batch_step_size': None,
				'loss': 'ce',
				'class_weights': class_weights,
				'opt': {
					'name': 'adam',
					'kwargs': {
						'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True)
					}
				},
				'num_workers': 0,
				'pin_memory': True
			}
		else:
			params = {
				'window_size': WIN_SIZE,
				'feat_dim': None,
				'train_shuffle': False,
				'epochs': 200,
				'batch_size': 128,
				'batch_step_size': None,
				'loss': 'ce',
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

