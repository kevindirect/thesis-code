"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK

from common_util import MODEL_DIR, window_iter
from model.common import MODELS_DIR, ERROR_CODE


class SequentialMixin:
	"""
	A Mixin for sequential models like RNNs.
	"""

	def preproc(self, params, data):
		"""
		Reshaping transform for sequential data.
		Resamples features to shape (samples, timesteps, features)
		"""
		# Reshape features into overlapping moving window samples
		f = np.stack(window_iter(data[0], n=params['step_size']))

		# Drop lables prior to the first step
		l = data[1][params['step_size']-1:]

		return f, l
