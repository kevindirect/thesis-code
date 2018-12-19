"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK
from keras import losses

from common_util import MODEL_DIR, in_debug_mode
from model.common import MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.model.Classifier import Classifier
from recon.split_util import get_train_test_split


class BinaryClassifier(Classifier):
	"""Abstract Base Class of all binary classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'output_activation' : hp.choice('output_activation', ['sigmoid', 'exponential', 'elu', 'tanh']),
			#'loss': hp.choice('loss', ['binary_crossentropy', 'hinge', 'squared_hinge', 'kullback_leibler_divergence'])
			'loss': hp.choice('loss', ['binary_crossentropy'])	# Setting to binary cross entropy loss only
		}
		super(BinaryClassifier, self).__init__({**default_space, **other_space})

	def make_const_data_objective(self, features, labels, loss_type='val_loss', retain_holdout=True, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, shuffle=False):
		"""
		Return an objective function that hyperopt can use for the given features and labels.
		"""
		feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, labels, test_ratio=test_ratio, shuffle=shuffle)

		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			try:
				compiled = self.get_model(params, features.shape[1])

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

				return {'loss': results['history'][loss_type][-1], 'status': STATUS_OK}

			except:
				self.bad_trials += 1
				return {'loss': ERROR_CODE, 'status': STATUS_OK}

		return objective