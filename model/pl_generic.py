"""
Kevin Patel
"""
import sys
import os
import logging
import pickle
from inspect import getfullargspec

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl # PL ver 1.2.3
import torchmetrics as tm

from common_util import load_df, dump_df, is_valid, isnt, rectify_json, dump_json
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_LOSS_MAPPING, PYTORCH_OPT_MAPPING, PYTORCH_SCH_MAPPING
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
	def __init__(self, pt_model_fn, params_m, params_t, fshape, splits=('train', 'val')):
		"""
		Init method

		Args:
			pt_model_fn (function): pytorch model callback
			params_m (dict): dictionary of model hyperparameters
			params_t (dict): dictionary of training hyperparameters
			fshape (tuple): the shape of a single feature observation,
				this is usually the model input shape
			splits (tuple): which splits to init metric objects for
		"""
		super().__init__()
		self.name = f'{self._get_name()}_{pt_model_fn.__name__}'
		self.params_m, self.params_t = params_m, params_t
		self.__init_loss_fn__()
		self.__init_model__(pt_model_fn, fshape)
		self.__init_loggers__(splits)
		# self.__init_loggers_return__(splits)
		#self.example_input_array = torch.rand(10, *fshape, dtype=torch.float32) * 100

	def __init_loss_fn__(self, reduction='none'):
		if (is_valid(loss_fn := PYTORCH_LOSS_MAPPING.get(self.params_t['loss'], None))):
			if (is_valid(cw := self.params_t['class_weights'])):
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
		model_params = {k: v for k, v in self.params_m.items() \
			if (k in getfullargspec(pt_model_fn).args)}

		self.model = pt_model_fn(in_shape=fshape, **model_params)
		if (is_valid(self.params_m.get('ob_out_shapes', None))):
			if (isnt(self.params_m.get('ob_params', None))):
				logging.info('using default output block params')
			self.model = OutputBlock.wrap(self.model)

		self.model_type = self.params_t['loss'].split('-')[0]

	def __init_loggers__(self, splits):
		"""
		'micro' weights by class frequency, 'macro' weights classes equally
		"""
		self.epoch_returns = None
		if (self.model_type == 'clf'):
			if (self.params_t['loss'] in ('clf-ce', 'clf-nll')):
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
			if (self.params_t['loss'] in ('reg-sharpe',)):
				num_classes = self.params_m['num_classes'] or self.params_m['out_size'] + 1
			else:
				assert(isnt(self.params_m['num_classes']))
			self.epoch_metrics = {
				epoch_type: {
					f'{self.model_type}_mae': tm.MeanAbsoluteError(compute_on_step=False),
					f'{self.model_type}_mse': tm.MeanSquaredError(compute_on_step=False),
				}
				for epoch_type in splits
			}

	def __init_loggers_return__(self, splits, go_long=True, go_short=False):
		self.epoch_returns = {}
		for epoch_type in splits:
			epoch_ret = {}
			if (go_long and go_short):
				br = SimulatedReturn(use_conf=False, compounded=False,
					pred_type=self.model_type)
				epoch_ret[br.name] = br
			if (go_long):
				br_long = SimulatedReturn(use_conf=False, compounded=False,
					pred_type=self.model_type)
				epoch_ret[br_long.name+'_long'] = br_long
			if (go_short):
				br_short = SimulatedReturn(use_conf=False, compounded=False,
					pred_type=self.model_type)
				epoch_ret[br_short.name+'_short'] = br_short

			for thresh in [None,]: #, .050, .125, .250, .500, .750]:
				if (go_long and go_short):
					cr = SimulatedReturn(use_conf=True, use_kelly=False, compounded=False, \
						pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
					kr = SimulatedReturn(use_conf=True, use_kelly=True, compounded=False, \
						pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
					epoch_ret[cr.name] = cr
					epoch_ret[kr.name] = kr
				if (go_long):
					cr_long = SimulatedReturn(use_conf=True, use_kelly=False, compounded=False, \
						pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
					kr_long = SimulatedReturn(use_conf=True, use_kelly=True, compounded=False, \
						pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
					epoch_ret[cr_long.name+'_long'] = cr_long
					epoch_ret[kr_long.name+'_long'] = kr_long
				if (go_short):
					cr_short = SimulatedReturn(use_conf=True, use_kelly=False, compounded=False, \
						pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
					kr_short = SimulatedReturn(use_conf=True, use_kelly=True, compounded=False, \
						pred_type=self.model_type, dir_thresh=thresh, conf_thresh=thresh)
					epoch_ret[cr_short.name+'_short'] = cr_short
					epoch_ret[kr_short.name+'_short'] = kr_short
			self.epoch_returns[epoch_type] = epoch_ret

	def configure_optimizers(self):
		"""
		Construct and return optimizers
		"""
		opt_fn = PYTORCH_OPT_MAPPING.get(self.params_t['opt']['name'])
		opt = opt_fn(self.parameters(), **self.params_t['opt']['kwargs'])
		return opt
		#sch_fn = PYTORCH_SCH_MAPPING.get(self.params_t['sch']['name'])
		#sch = sch_fn(opt, **self.params_t['sch']['kwargs'])
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
			if (self.params_t['loss'] in ('clf-bce',)):
				pred_t_loss = pred_t_raw
				pred_t_loss = F.sigmoid(pred_t_loss, dim=-1)
				pred_t = pred_t_loss.detach().clone()
				pred_t_ret = (pred_t - .5) * 2
			elif (self.params_t['loss'] in ('clf-ce',)):
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
				if (name.endswith('long')):
					d = ret.compute(f'{epoch_type}_{name}',
						go_long=True, go_short=False)
				elif (name.endswith('short')):
					d = ret.compute(f'{epoch_type}_{name}',
						go_long=False, go_short=True)
				else:
					d = ret.compute(f'{epoch_type}_{name}',
						go_long=True, go_short=True)

				self.log_dict(d, prog_bar=False, logger=True,
					on_step=False, on_epoch=True)

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

	def compute_results_json(self):
		"""
		"""
		results_json = {}
		for split in self.epoch_metrics:
			split_results = {}
			for name in self.epoch_metrics[split]:
				em = self.epoch_metrics[split][name]
				split_results[f'{split}_{name}'] = em.compute()
			for name in self.epoch_returns[split]:
				er = self.epoch_returns[split][name]
				if (name.endswith('long')):
					d = er.compute(f'{split}_{name}', go_long=True, go_short=False)
				elif (name.endswith('short')):
					d = er.compute(f'{split}_{name}', go_long=False, go_short=True)
				else:
					d = er.compute(f'{split}_{name}', go_long=True, go_short=True)
				split_results.update(d)
			results_json[split] = rectify_json(split_results)
		return results_json

	def dump_plots_losscurve(self):
		pass

	# def dump_results(self, results_dir, model_name):
	# 	results_json = self.compute_results_json()
	# 	for split, result in results_json.items():
	# 		dump_json(result, split, results_dir)
	# 	return results_json

	# def dump_plots_return(self, plot_dir, model_name, dm):
	# 	"""
	# 	"""
	# 	bothdir = lambda n: not (n.endswith('long') or n.endswith('short'))

	# 	for split in self.epoch_returns:
	# 		# for name in filter(bothdir, self.epoch_returns[split]):
	# 		for name in self.epoch_returns[split]:
	# 			er = self.epoch_returns[split][name]
	# 			plot_name = model_name.upper() +f"-{name}".title()
	# 			fig, axes = er.plot_result_series(split.title(),
	# 				plot_name, dm.idx[split])
	# 			fname = f"{split}_{plot_name}".lower()
	# 			# with open(f'{plot_dir}{fname}.pickle', 'wb') as f:
	# 			# 	pickle.dump(fig, f)
	# 			plt.savefig(f'{plot_dir}{fname}', bbox_inches="tight",
	# 				transparent=True)
	# 			plt.close(fig)

	# 		# er = self.epoch_returns[split][name]
	# 		# plot_name = model_name.upper() +f"-{name}".title()
	# 		# fig, axes = er.plot_result_series(split.title(),
	# 		# 	plot_name, dm.idx[split])
	# 		# fname = f"{split}_{plot_name}".lower()
	# 		# plt.savefig(f'{plot_dir}{fname}', bbox_inches="tight",
	# 		# 	transparent=True)
	# 		# plt.close(fig)

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

