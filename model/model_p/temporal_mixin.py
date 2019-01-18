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
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE


class TemporalMixin:
	"""
	A Mixin for models which look at the entire effective history of a data point at once.
	This is in contrast to models like RNNs where each data point may be a group of timesteps fed in sequence.
	"""

	def preproc(self, params, data):
		"""
		Reshaping transform for temporal data.

		Runs a "moving window unstack" operation through the first data such that each row of the result contains the history
		of the original up to and including that row based on a num_windows parameter in params. The num_windows
		determines how far back the history each row will record; a num_windows of '1' results in no change. 
		
		example with num_windows of '2':
													0 | a b c 
													1 | d e f ---> 1 | a b c d e f
													2 | g h i      2 | d e f g h i
													3 | j k l      3 | g h i j k l

		All data after the first tuple item are assumed to be label/target vectors and are reshaped to align with the new first
		tuple item.
		"""
		# Reshape features into overlapping moving window samples
		f = np.array([np.concatenate(vec) for vec in window_iter(data[0], n=params['num_windows'])])

		# Drop lables prior to the first step for label/target vectors
		l = tuple(label[params['num_windows']-1:] for label in data[1:])

		return (f, *l)
