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
import tensorflow as tf

from common_util import MODEL_DIR, identity_fn
from model.common import TENSORFLOW_MODEL_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO


class Model:
	"""
	Abstract base class of all pure tensorflow based model subclasses.
	Models bundle a supervised learning model with a hyperopt parameter space to search over.
	Models do not store any experiment or trial information, instead they specify the model structure and parameter space. In this
	way they are more like model factories than model objects.
	"""
	def __init__(self, other_space={}):
		default_space = {
			'epochs': hp.choice('epochs', [200]),
			'batch_size': hp.choice('batch_size', [64, 128, 256])
		}
		self.space = {**default_space, **other_space}

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

	def fit_model(self, params, logdir, model, train_data, val_data=None, val_split=VAL_RATIO, shuffle=False):
		"""
		Fit the model to training data and return results on the validation data.
		"""
		try:
			

		except Exception as e:
			logging.error('Error during model fitting: {}'.format(str(e)))
			raise e

		return

