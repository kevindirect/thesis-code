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
from model.model_t.classifier import Classifier
from recon.split_util import get_train_test_split


class BinaryClassifier(Classifier):
	"""Abstract Base Class of all binary classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'loss': hp.choice('loss', ['sparse_ce'])
		}
		super(BinaryClassifier, self).__init__({**default_space, **other_space})
