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
from keras.optimizers import SGD, RMSprop, Adam, Nadam

from common_util import MODEL_DIR
from model.common import MODELS_DIR, ERROR_CODE
from recon.split_util import get_train_test_split


class ClassifierExperiment:
	"""
	Abstract Class of all classifier experiments.
	Classifier experiments bundle a keras classifier with a hyperopt parameter space to search over.
	"""

	def __init__(self, other_space={}):
		default_space = {
			'opt': hp.choice('opt', [SGD, RMSprop, Adam, Nadam]),
			'lr': hp.choice('lr', [0.01, 0.02, 0.001, 0.0001]),
			'epochs': hp.choice('epochs', [5, 20, 50, 100]),
			'batch_size': hp.choice('batch_size', [64, 128, 256])
		}
		self.space = {**default_space, **other_space}
		self.metrics = ['accuracy']
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
		Define, compile, and return a keras model over params.
		"""
		pass

	def fit_model(self, params, model, train_data, val_data=None, val_split=.25, shuffle=False):
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

	def make_const_data_objective(self, features, labels, retain_holdout=True, test_ratio=.25, shuffle=False):
		"""
		Return an objective function that hyperopt can use for the given features and labels.
		"""
		binary_case = False if (labels.unique() > 2) else True

		if (binary_case):
			feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, labels, test_ratio=test_ratio, shuffle=shuffle)
		else:
			feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, pd.get_dummies(labels, drop_first=False), test_ratio=test_ratio, shuffle=shuffle)

		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			try:
				compiled = self.make_model(params, (features.shape[1],))

				if (retain_holdout):
					results = self.fit_model(params, compiled, (feat_train, lab_train), val_data=None, val_split=test_ratio, shuffle=shuffle)
				else:
					results = self.fit_model(params, compiled, (feat_train, lab_train), val_data=(feat_test, lab_test), val_split=test_ratio, shuffle=shuffle)

				return {'loss': results['history']['val_loss'][-1], 'status': STATUS_OK}

			except:
				self.bad_trials += 1
				logging.error('Error ocurred during experiment')
				return {'loss': ERROR_CODE, 'status': STATUS_OK}

		return objective

	def make_var_data_objective(self, raw_features, raw_labels, features_fn, labels_fn, retain_holdout=True, test_ratio=.25, shuffle=False):
		"""
		Return an objective function that hyperopt can use that can search over features and labels along with the hyperparameters.
		"""
		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			features, labels = features_fn(raw_features, params), labels_fn(raw_labels, params)
			return self.make_const_data_objective(features, labels, retain_holdout=retain_holdout, test_ratio=test_ratio, shuffle=shuffle)

		return objective

