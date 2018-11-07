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
from model.common import MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.model.Model import Model
from recon.split_util import get_train_test_split


class Classifier(Model):
	"""Abstract Base Class of all classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'opt': hp.choice('opt', [SGD, RMSprop, Adam, Nadam]),
			'lr': hp.choice('lr', [0.01, 0.02, 0.001, 0.0001])
		}
		super(Classifier, self).__init__({**default_space, **other_space})
		self.metrics = ['accuracy']

	def make_const_data_objective(self, features, labels, retain_holdout=True, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, shuffle=False):
		"""
		Return an objective function that hyperopt can use for the given features and labels.
		"""
		if (labels.unique().size > 2):
			labels = pd.get_dummies(labels, drop_first=False) # If the labels are not binary (more than two value types), one hot encode them
		
		feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, labels, test_ratio=test_ratio, shuffle=shuffle)

		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			try:
				compiled = self.make_model(params, (features.shape[1],))

				if (retain_holdout):
					results = self.fit_model(params, compiled, (feat_train, lab_train), val_data=None, val_split=val_ratio, shuffle=shuffle)
				else:
					results = self.fit_model(params, compiled, (feat_train, lab_train), val_data=(feat_test, lab_test), val_split=val_ratio, shuffle=shuffle)

				if (in_debug_mode()):
					val_loss, val_acc = results['history']['val_loss'], results['history']['val_acc']
					logging.debug('val_loss mean, min, max, last: {mean}, {min}, {max}, {last}'
						.format(mean=np.mean(val_loss), min=np.min(val_loss), max=np.max(val_loss), last=val_loss[-1]))
					logging.debug('val_acc mean, min, max, last: {mean}, {min}, {max}, {last}'
						.format(mean=np.mean(val_acc), min=np.min(val_acc), max=np.max(val_acc), last=val_acc[-1]))

				return {'loss': results['history']['val_loss'][-1], 'status': STATUS_OK}

			except:
				self.bad_trials += 1
				logging.error('Error ocurred during experiment')
				return {'loss': ERROR_CODE, 'status': STATUS_OK}

		return objective

	def make_var_data_objective(self, raw_features, raw_labels, features_fn, labels_fn, retain_holdout=True, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, shuffle=False):
		"""
		Return an objective function that hyperopt can use that can search over features and labels along with the hyperparameters.
		"""
		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			features, labels = features_fn(raw_features, params), labels_fn(raw_labels, params)
			return self.make_const_data_objective(features, labels, retain_holdout=retain_holdout, test_ratio=test_ratio, val_ratio=val_ratio, shuffle=shuffle)

		return objective

