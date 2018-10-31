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

from common_util import MODEL_DIR
from model.common import MODELS_DIR, ERROR_CODE
from model.models.ClassifierExperiment import ClassifierExperiment
from recon.split_util import get_train_test_split


class BinaryClassifierExperiment(ClassifierExperiment):
	"""Abstract Base Class of all binary classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'output_activation' : hp.choice('output_activation', ['sigmoid', 'exponential', 'elu', 'tanh']),
			'loss': hp.choice('loss', [losses.binary_crossentropy, losses.hinge, losses.squared_hinge, losses.kullback_leibler_divergence])
		}
		super(BinaryClassifierExperiment, self).__init__({**default_space, **other_space})

	def make_const_data_objective(self, features, labels, retain_holdout=True, test_ratio=.25, shuffle=False):
		"""
		Return an objective function that hyperopt can use for the given features and labels.
		"""
		feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, labels, test_ratio=test_ratio, shuffle=shuffle)

		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			try:
				compiled = self.make_model(params, features.shape[0])

				if (retain_holdout):
					results = self.fit_model(params, compiled, feat_train, lab_train, val_split=test_ratio, shuffle=shuffle)
				else:
					results = self.fit_model(params, compiled, feat_train, lab_train, lab_train, lab_test, val_split=test_ratio, shuffle=shuffle)

				return {'loss': results, 'status': STATUS_OK}

			except:
				logging.error('Error ocurred during experiment')
				return {'loss': ERROR_CODE, 'status': STATUS_OK}

		return objective