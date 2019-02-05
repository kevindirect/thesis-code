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
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.model_p.classifier import Classifier


class BinaryClassifier(Classifier):
	"""
	Abstract Base Class of all binary classifiers.

	Parameters:
		loss (string): string representing pytorch loss to use
	"""

	def __init__(self, other_space={}):
		default_space = {
			'loss': hp.choice('loss', ['ce'])
		}
		# default_space = {
		# 	'loss': hp.choice('loss', ['bce', 'ce', 'bcel', 'nll'])
		# }
		super(BinaryClassifier, self).__init__({**default_space, **other_space})
