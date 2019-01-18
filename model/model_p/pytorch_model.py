"""
Kevin Patel
"""
import sys
import os
from os import sep
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter

from common_util import MODEL_DIR, identity_fn
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO


class Model:
	"""
	Abstract base class of all pytorch based model subclasses.
	Models bundle a supervised learning model with a hyperopt parameter space to search over.
	Models do not store any experiment or trial information, instead they specify the model structure and parameter space;
	They are more like model factories than model objects.

	Parameters:
		epochs (int > 0): epochs to train model
		batch_size (int > 0): batch sized used in training
	"""
	def __init__(self, other_space={}):
		default_space = {
			'epochs': hp.choice('epochs', [200]),
			'batch_size': hp.choice('batch_size', [64, 128, 256])
		}
		self.space = {**default_space, **other_space}
		self.tbx = lambda params, logdir: SummaryWriter(logdir) # Creates TensorBoardX logger
		self.metrics_fns = {
			'accuracy': accuracy_score,
			'precision': precision_score,
			'recall': recall_score,
			'f1': f1_score
		}

	def get_space(self):
		return self.space

	def params_idx_to_name(self, params_idx):
		"""
		Loop over params_idx dictionary and map indexes to parameter values in hyperspace dictionary.
		"""
		params_dict = {}

		for name, idx in params_idx.items():
			hp_obj = self.space[name]
			hp_obj_type = hp_obj.name

			if (hp_obj_type == 'switch'): # Indicates an hp.choice object
				choice_list = hp_obj.pos_args[1:]
				chosen = choice_list[idx]._obj
				if (isinstance(chosen, str) or isinstance(chosen, int) or isinstance(chosen, float)):
					params_dict[name] = chosen
				else:
					try:
						params_dict[name] = chosen.__name__
					except:
						params_dict[name] = str(chosen)

			elif (hp_obj_type == 'float'): # Indicates a hp sampled value
				params_dict[name] = idx

		return params_dict

	def preproc(self, params, data):
		"""
		Apply any final transforms or reshaping to passed data tuple before fitting.
		"""
		return identity_fn(data)

	def batchify(self, params, data, device, shuffle_batches=False):
		"""
		Takes in final numpy data and returns torch DataLoader over torch tensor minibatches of specified torch device.
		"""
		ds = TensorDataset(*[torch.tensor(d, device=device) for d in data])
		dl = DataLoader(ds, batch_size=params['batch_size'], shuffle=shuffle_batches)
		return dl

	def batch_loss_metrics(model, loss_function, feat_batch, lab_batch, optimizer=None):
		"""
		Compute loss and metrics on batch, run optimizer on losses if passed.
		"""
		predictions = model(feat_batch)
		loss = loss_function(predictions, lab_batch)
		metrics = {name: fn(lab_batch, predictions) for name, fn in self.metrics_fns}

		if (optimizer is not None):
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		return loss.item(), len(feat_batch), metrics

	def make_model(self, params, num_inputs):
		"""
		Define, compile, and return a model over params.
		Concrete model subclasses must implement this.
		"""
		pass

	def get_model(self, params, num_inputs):
		"""
		Wrapper around make_model that reports/handles errors.
		"""
		try:
			model = self.make_model(params, num_inputs)

		except Exception as e:
			logging.error('Error during model creation: {}'.format(str(e)))
			raise e

		return model

	def fit_model(self, params, logdir, model, device, train_data, val_data=None):
		"""
		Fit the model to training data and return results on the validation data.
		"""
		try:
			history = {
				'loss': [],
				'val_loss': []
			}
			loss_fn, opt = self.make_loss_fn(params), self.make_optimizer(params)
			writer = self.tbx(logdir)
			
			for epoch in range(params['epochs']):
				model.train()
				for Xb, yb in self.batchify(params, self.preproc(params, train_data), device, shuffle_batches=True):
					losses, nums, metrics = self.batch_loss_metrics(model, loss_fn, Xb, yb, optimizer=opt)
				loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
				history['loss'].append(loss)
				writer.add_scalar('data/train/loss', loss, epoch)
				writer.add_scalars('data/train/metrics', metrics, epoch)

				model.eval()
				with torch.no_grad():
					losses, nums, metrics = zip(*[self.batch_loss_metrics(model, loss_fn, Xb, yb) for Xb, yb in self.batchify(params, self.preproc(params, val_data), device, False)])
				loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
				history['val_loss'].append(loss)
				writer.add_scalar('data/val/loss', loss, epoch)
				writer.add_scalars('data/val/metrics', metrics, epoch)

			writer.export_scalars_to_json('results.json')
			writer.close()

			results = {
				# 'history': history
				'mean': {
					'loss': np.mean(history['loss']),
					'val_loss': np.mean(history['val_loss'])
				},
				'last': {
					'loss': history['loss'][-1],
					'val_loss': history['val_loss'][-1]
				}
			}

		except Exception as e:
			logging.error('Error during model fitting: {}'.format(str(e)))
			raise e

		return results
