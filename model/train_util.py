"""
Kevin Patel
"""
import sys
import os
from os import sep
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from common_util import MODEL_DIR, identity_fn, is_type, is_ser, isnt, np_inner, get0, midx_split, pd_midx_to_arr
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO


def pd_get_np_tvt(pd_obj, as_midx=True, train_ratio=.6):
	"""
	Return the train, val, test numpy splits of a pandas object

	Args:
		pd_obj (pd.DataFrame|pd.Series): data to split
		as_midx (bool): whether to interpret as MultiIndex indexed object
		train_ratio (float (0,1)): train ratio, remainder is equally split among val/test

	Returns:
		split data as numpy arrays
	"""
	tv_ratio = (1-train_ratio)/2
	train_idx, val_idx, test_idx = midx_split(pd_obj.index, train_ratio, tv_ratio, tv_ratio)
	train_pd, val_pd, test_pd = pd_obj.loc[train_idx], pd_obj.loc[val_idx], pd_obj.loc[test_idx]

	if (as_midx and is_type(pd_obj.index, pd.core.index.MultiIndex)):
		train_np, val_np, test_np = map(pd_midx_to_arr, map(lambda d: d if (is_ser(d)) else d.stack(), [train_pd, val_pd, test_pd]))
	else:
		train_np, val_np, test_np = train_pd.values, val_pd.values, test_pd.values
	return train_np, val_np, test_np


def batchify(params, data, shuffle_batches=False):
	"""
	Return a torch.DataLoader made from a tuple of numpy arrays.

	Args:
		params (dict): model parameters dictionary
		data (tuple): tuple of numpy arrays, features are the first element
		shuffle_batches (bool): whether or not to shuffle the batches

	Returns:
		torch.DataLoader
	"""
	f = torch.tensor(data[0], dtype=torch.float32, requires_grad=True)
	if (params['loss'] in ('bce', 'bcel', 'mae', 'mse')):
		l = [torch.tensor(d, dtype=torch.float32, requires_grad=False) for d in data[1:]]
	elif (params['loss'] in ('ce', 'nll')):
		l = [torch.tensor(d, dtype=torch.int64, requires_grad=False).squeeze() for d in data[1:]]
	ds = TensorDataset(f, *l)
	dl = DataLoader(ds, batch_size=params['batch_size'], shuffle=shuffle_batches)
	return dl


def batch_output(params, model, loss_fn, feat_batch, lab_batch):
	"""
	Run batch on model, return output batch and loss.

	Args:
		params (dict): model parameters dictionary
		model (nn.module): torch model
		loss_fn: torch loss function
		feat_batch: feature batch
		lab_batch: label/target batch

	Returns:
		output batch and loss
	"""
	output_batch = model(feat_batch)
	loss = loss_fn(output_batch, lab_batch)
	return output_batch, loss


def batch_metrics(params, output_batch, lab_batch, metrics_fn):
	"""
	Make predictions from output batch and run over metrics.

	Args:
		params (dict): model parameters dictionary
		output_batch (): torch model batch outputs
		lab_batch (): label/target batch
		metrics_fns (dict): dictionary of metric functions
		optimizer
		ret_train_pred

	Returns:

	"""
	max_batch, pred_batch = torch.max(outputs_batch, dim=1) # Convert network outputs into predictions
	lab_batch_cpu = lab_batch.cpu()
	pred_batch_cpu = pred_batch.cpu()
	metrics = {name: fn(lab_batch_cpu, pred_batch_cpu) for name, fn in metrics_fns.items()}


def model_step(loss, opt):
	opt.zero_grad()
	loss.backward()
	opt.step()
	if (not ret_train_pred):
		return loss.item(), len(feat_batch), metrics

	# logging.debug('batch loss:   {}'.format(loss.item()))
	return loss.item(), len(feat_batch), metrics, (max_batch.exp(), pred_batch.float())


def fit_model(self, params, logdir, metrics_fns, model, device, train_data, val_data=None):
	"""
	Fit the model to training data and return results on the validation data.
	"""
	try:
		history = {
			'loss': [],
			'val_loss': []
		}
		for name in metrics_fns.keys():
			history[name] = []
			history['val_{}'.format(name)] = []

		loss_fn, opt = self.make_loss_fn(params).to(device), self.make_optimizer(params, model.parameters())
		writer = self.tbx(params, logdir) if (logdir is not None) else None
		model.zero_grad()
		opt.zero_grad()

		logging.debug('w[-2:][-2:]: {}'.format(list(model.parameters())[-2:][-2:]))

		for epoch in range(params['epochs']):
			epoch_str = str(epoch).zfill(3)
			model.train()
			losses, nums, metrics = zip(*[batch_loss(params, model, loss_fn, Xb, yb, optimizer=opt) for Xb, yb in batchify(params, self.preproc(params, train_data), device, shuffle_batches=True)])
			# for Xb, yb in self.batchify(params, self.preproc(params, train_data), device, shuffle_batches=True):
			# 	losses, nums, metrics = self.batch_loss(params, model, loss_fn, Xb, yb, optimizer=opt)
			loss = np_inner(losses, nums)
			soa = {name[0]: tuple(d[name[0]] for d in metrics) for name in zip(*metrics)}
			metric = {name: np_inner(vals, nums) for name, vals in soa.items()}

			logging.debug('{} train loss: {}'.format(epoch_str, loss))
			history['loss'].append(loss)
			for name, val in metric.items():
				history[name].append(val)

			if (writer is not None):
				writer.add_scalar('data/train/loss', loss, epoch)
				writer.add_scalars('data/train/metric', metric, epoch)

			logging.debug('{} w[-2:][-2:]: {}'.format(epoch_str, list(model.parameters())[-2:][-2:]))

			model.eval()
			with torch.no_grad():
				Xe, ye = get0(*batchify(params, self.preproc(params, val_data), device, override_batch_size=val_data[-1].size, shuffle_batches=False))
				loss, num, metric, pred = batch_loss(params, model, loss_fn, Xe, ye)

			logging.debug('{} val loss: {}'.format(epoch_str, loss))
			history['val_loss'].append(loss)
			for name, val in metric.items():
				history['val_{}'.format(name)].append(val)

			if (writer is not None):
				writer.add_scalar('data/val/loss', loss, epoch)
				writer.add_scalars('data/val/metric', metric, epoch)

			logging.debug('{} w[-2:][-2:]: {}'.format(epoch_str, list(model.parameters())[-2:][-2:]))

		if (writer is not None):
			writer.export_scalars_to_json(logdir +'results.json')
			writer.close()

		results = {
			'history': history,
			'mean': {name: np.mean(vals) for name, vals in history.items()},
			'last': {name: vals[-1] for name, vals in history.items()}
		}

	except Exception as e:
		logging.exception('Error during model fitting: {}'.format(str(e)))
		raise e

	return results

