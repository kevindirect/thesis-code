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

from common_util import MODEL_DIR, identity_fn, is_type
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
			'batch_size': hp.choice('batch_size', [128, 256])
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

	# def params_idx_to_name(self, params_idx):
	# 	def translate_param_idx(hp_space, params_idx):
	# 		def handle_param(hp_param_obj, hp_type, hp_idx, param_name, res):
	# 			if (hp_type == 'switch'):			# Indicates an hp.choice object
	# 				choice_list = hp_param_obj.pos_args[1:]
	# 				chosen = choice_list[hp_idx]
					
	# 				if (len(chosen.named_args) > 0): # Nested hp.choice
	# 					for subparam in chosen.named_args:
	# 						if (is_type(subparam[0], str)):
	# 							sp_name = '_'.join([param_name, subparam[0]])
	# 							sp_obj = subparam[1]
	# 							print('dir(sp_obj)', dir(sp_obj))
	# 							handle_param(subparam[1], sp_obj, 0, sp_name, res)
					
	# 				chosen_value = chosen.obj
	# 				if (is_type(chosen, bool, str, int, float)):
	# 					res[param_name] = chosen_value

	# 			elif (hp_type == 'float'):			# Indicates a hp sampled value
	# 				res[param_name] = hp_idx

	# 			return res

	# 		result = {}

	# 		for name, idx in params_idx.items():
	# 			if (name in hp_space):
	# 				hp_obj = hp_space[name]
	# 				handle_param(hp_obj, hp_obj.name, idx, name, result)

	# 		return result

	# 	return translate_param_idx(self.space, params_idx)

	def params_idx_to_name(self, params_idx):
		"""
		Loop over params_idx dictionary and map indexes to parameter values in hyperspace dictionary.
		"""
		params_dict = {}

		for name, idx in params_idx.items():
			if (name in self.space):
				hp_obj = self.space[name]
				hp_obj_type = hp_obj.name

				if (hp_obj_type == 'switch'): # Indicates an hp.choice object
					choice_list = hp_obj.pos_args[1:]
					chosen = choice_list[idx].obj

					if (is_type(chosen, bool, str, int, float)):
						params_dict[name] = chosen
					else:
						try:
							params_dict[name] = chosen.__name__
						except:
							params_dict[name] = str(chosen)

				elif (hp_obj_type == 'float'): # Indicates a hp sampled value
					params_dict[name] = idx
			else:
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
		f = torch.tensor(data[0], dtype=torch.float32, device=device, requires_grad=True)
		if (params['loss'] in ['bce', 'bcel']):
			l = [torch.tensor(d, dtype=torch.float32, device=device, requires_grad=False) for d in data[1:]]
		elif (params['loss'] in ['ce', 'nll']):
			l = [torch.tensor(d, dtype=torch.int64, device=device, requires_grad=False).squeeze() for d in data[1:]]
		ds = TensorDataset(f, *l)
		dl = DataLoader(ds, batch_size=params['batch_size'], shuffle=shuffle_batches)
		return dl

	def batch_loss(self, params, model, loss_function, feat_batch, lab_batch, optimizer=None):
		"""
		Compute loss and metrics on batch, run optimizer on losses if passed.
		"""
		# logging.debug('batch tensor[0][0]: {}'.format(feat_batch[0][0]))
		outputs_batch = model(feat_batch)
		loss = loss_function(outputs_batch, lab_batch)
		prediction_batch = torch.argmax(outputs_batch, dim=1) # Convert network outputs into predictions
		print(lab_batch.shape)
		print(prediction_batch.shape)
		metrics = {name: fn(lab_batch, prediction_batch) for name, fn in self.metrics_fns}

		if (optimizer is not None):
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		logging.debug('batch loss:   {}'.format(loss.item()))

		return loss.item(), len(feat_batch), metrics

	def make_model(self, params, input_shape):
		"""
		Define, compile, and return a model over params.
		Concrete model subclasses must implement this.
		"""
		pass

	def get_model(self, params, input_shape):
		"""
		Wrapper around make_model that reports/handles errors.
		"""
		try:
			model = self.make_model(params, input_shape)
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
			loss_fn, opt = self.make_loss_fn(params).to(device), self.make_optimizer(params, model.parameters())
			writer = self.tbx(params, logdir) if (logdir is not None) else None
			model.zero_grad()
			opt.zero_grad()

			logging.debug('w[-2:][-2:]: {}'.format(list(model.parameters())[-2:][-2:]))

			for epoch in range(params['epochs']):
				epoch_str = str(epoch).zfill(3)
				model.train()
				losses, nums, metrics = zip(*[self.batch_loss(params, model, loss_fn, Xb, yb, optimizer=opt) for Xb, yb in self.batchify(params, self.preproc(params, train_data), device, shuffle_batches=True)])
				# for Xb, yb in self.batchify(params, self.preproc(params, train_data), device, shuffle_batches=True):
				# 	losses, nums, metrics = self.batch_loss(params, model, loss_fn, Xb, yb, optimizer=opt)
				loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
				logging.debug('{} train loss: {}'.format(epoch_str, loss))
				history['loss'].append(loss)
				if (writer is not None):
					writer.add_scalar('data/train/loss', loss, epoch)
					writer.add_scalars('data/train/metrics', metrics, epoch)

				logging.debug('{} w[-2:][-2:]: {}'.format(epoch_str, list(model.parameters())[-2:][-2:]))

				model.eval()
				with torch.no_grad():
					losses, nums, metrics = zip(*[self.batch_loss(params, model, loss_fn, Xb, yb) for Xb, yb in self.batchify(params, self.preproc(params, val_data), device, shuffle_batches=False)])
				loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
				logging.debug('{} val loss: {}'.format(epoch_str, loss))
				history['val_loss'].append(loss)
				if (writer is not None):
					writer.add_scalar('data/val/loss', loss, epoch)
					writer.add_scalars('data/val/metrics', metrics, epoch)

				logging.debug('{} w[-2:][-2:]: {}'.format(epoch_str, list(model.parameters())[-2:][-2:]))

			if (writer is not None):
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

