"""
Kevin Patel
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK

from common_util import MODEL_DIR
from model.common import MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO


class Model:
	"""
	Abstract base class of all model subclasses.
	Models bundle a supervised learning model with a hyperopt parameter space to search over.
	"""
	def __init__(self, other_space={}):
		default_space = {
			'epochs': hp.choice('epochs', [5, 20, 50, 100]),
			'batch_size': hp.choice('batch_size', [64, 128, 256])
		}
		self.space = {**default_space, **other_space}
		self.history = History()
		# self.reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, verbose=1, mode='min', min_lr=0.000001) # TODO - try this out on fit function
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

	def make_model(self, params, input_shape):
		"""
		Define, compile, and return a model over params.
		"""
		pass

	def fit_model(self, params, model, train_data, val_data=None, val_split=VAL_RATIO, shuffle=False):
		"""
		Fit the model and return a dictionary describing the training and test results.
		"""
		stats = model.fit(*train_data, 
						epochs=params['epochs'], 
						batch_size=params['batch_size'], 
						callbacks=[self.history],
						verbose=1, 
						validation_split=val_split, # Overriden if validation data is not None
						validation_data=val_data if (val_data is not None) else None, 
						shuffle=shuffle)

		return {
			'model': stats.model,
			'params': stats.params,
			'val_data': stats.validation_data,
			'history': stats.history
		}
