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
import pytorch_lightning as pl # PL ver 1.2.3
import torchmetrics as tm

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
			if (isnt(self.m_params.get('ob_params', None))):
				logging.info('using default output block params')
			self.model = OutputBlock.wrap(self.model)

		self.model_type = self.t_params['loss'].split('-')[0]

	def __init_loggers__(self, epoch_metric_types):
		"""
		'micro' weights by class frequency, 'macro' weights classes equally
		"""
		if (self.model_type == 'clf'):
			if (self.t_params['loss'] in ('clf-ce',)):
				num_classes = self.m_params['label_size'] + 1
			else:
				num_classes = self.m_params['label_size']
			self.epoch_metrics = {
				epoch_type: {
					'accuracy': tm.Accuracy(compute_on_step=False),
					'precision': tm.Precision(num_classes=num_classes,
						average='micro', compute_on_step=False),
					'recall': tm.Recall(num_classes=num_classes,
						average='micro', compute_on_step=False),
					'f1.0': tm.FBeta(num_classes=num_classes, beta=1.0,
						average='micro', compute_on_step=False),
					# 'f0.5': tm.FBeta(num_classes=num_classes, beta=0.5,
					# 	average='micro', compute_on_step=False),
				}
				for epoch_type in epoch_metric_types
			}
		elif (self.model_type == 'reg'):
			self.epoch_metrics = {
				epoch_type: {
					'mae': tm.MeanAbsoluteError(compute_on_step=False),
					'mse': tm.MeanSquaredError(compute_on_step=False),
				}
				for epoch_type in epoch_metric_types
			}

		self.epoch_returns = {}
		for epoch_type in epoch_metric_types:
			epoch_ret = {}
			br = SimulatedReturn(use_conf=False, compounded=False, pred_type=self.model_type)
			epoch_ret[br.name] = br

			for thresh in [None, .050, .125, .250, .500, .750]:
				cr = SimulatedReturn(use_conf=True, use_kelly=False, compounded=False, \
					pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
				kr = SimulatedReturn(use_conf=True, use_kelly=True, compounded=False, \
					pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
				epoch_ret[cr.name] = cr
				epoch_ret[kr.name] = kr

			# for thresh in [.500,]:
			# 	cr = SimulatedReturn(use_conf=True, use_kelly=False, compounded=False, \
			# 		pred_type=self.model_type, dir_thresh=thresh)
			# 	kr = SimulatedReturn(use_conf=True, use_kelly=True, compounded=False, \
			# 		pred_type=self.model_type, dir_thresh=thresh)
			# 	epoch_ret[cr.name] = cr
			# 	epoch_ret[kr.name] = kr

			self.epoch_returns[epoch_type] = epoch_ret

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
			if (self.t_params['loss'] in ('clf-ce',)):
				pred_t_loss = pred_t_raw
				pred_t_smax = F.softmax(pred_t_raw.detach().clone(), dim=-1)
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

		if (is_valid(self.epoch_returns)):
			for ret in self.epoch_returns[epoch_type].values():
				try:
					ret.update(pred_ret.cpu(), z.cpu())
				except Exception as err:
					print("Error! pl_generic.py > GenericModel > forward_step() > ret.update()\n",
						sys.exc_info()[0], err)
					print(f'{ret=}')
					print(f'{pred_ret.shape=}')
					print(f'{z.shape=}')
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

		if (is_valid(self.epoch_returns)):
			for name, ret in self.epoch_returns[epoch_type].items():
				self.log_dict(ret.compute(pfx=epoch_type), prog_bar=False, \
					logger=True, on_step=False, on_epoch=True)

	def reset_metrics(self, epoch_type):
		"""
		Reset/Clear the running metrics.
		"""
		for name, met in self.epoch_metrics[epoch_type].items():
			met.reset()

		if (is_valid(self.epoch_returns)):
			for name, ret in self.epoch_returns[epoch_type].items():
				ret.reset()

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

