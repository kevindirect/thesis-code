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
from model.model.classifier import Classifier
from recon.split_util import get_train_test_split


class CategoricalClassifier(Classifier):
	"""Abstract Base Class of all categorical (multi class) classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'output_activation' : hp.choice('output_activation', ['softmax']),
			'loss': hp.choice('loss', ['categorical_crossentropy', 'sparse_categorical_crossentropy', 'categorical_hinge', 'cosine_proximity'])
		}
		super(CategoricalClassifier, self).__init__({**default_space, **other_space})
