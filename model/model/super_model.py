"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK
from keras.callbacks import Callback, BaseLogger, History, EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger, LambdaCallback

from common_util import MODEL_DIR, identity_fn
from model.common import MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO


class Model:
	"""
	Abstract base class of all model subclasses.
	Models bundle a supervised learning model with a hyperopt parameter space to search over.
	"""
	def __init__(self, other_space={}):
		default_space = {
			'epochs': hp.choice('epochs', [200]),
			'batch_size': hp.choice('batch_size', [64, 128, 256])
		}
		self.space = {**default_space, **other_space}
		self.callbacks = [History]
		# EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
		# ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, verbose=1, mode='min', min_lr=0.000001)
		self.bad_trials = 0

	def get_space(self):
		return self.space

	def get_bad_trials(self):
		return self.bad_trials

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

	def fit_model(self, params, model, train_data, val_data=None, val_split=VAL_RATIO, shuffle=False):
		"""
		Fit the model and return a dictionary describing the training and test results.
		"""
		try:
			stats = model.fit(*self.preproc(params, train_data), 
							epochs=params['epochs'], 
							batch_size=params['batch_size'], 
							callbacks=[init() for init in self.callbacks], 
							verbose=1, 
							validation_split=val_split, # Only used if validation_data is None
							validation_data=self.preproc(params, val_data) if (val_data is not None) else None, 
							shuffle=shuffle)

		except Exception as e:
			logging.error('Error during model fitting: {}'.format(str(e)))
			raise e

		return {
			'model': stats.model,
			'params': stats.params,
			'val_data': stats.validation_data,
			'history': stats.history
		}
