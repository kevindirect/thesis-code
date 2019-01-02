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
from model.common import MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.model_k.classifier import Classifier


class BinaryClassifier(Classifier):
	"""Abstract Base Class of all binary classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'output_activation' : hp.choice('output_activation', ['sigmoid', 'exponential', 'elu', 'tanh']),
			'loss': hp.choice('loss', ['binary_crossentropy'])	# Fix loss to binary cross entropy
		}
		super(BinaryClassifier, self).__init__({**default_space, **other_space})
