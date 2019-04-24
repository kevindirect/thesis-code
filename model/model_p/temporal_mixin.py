"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK

from common_util import MODEL_DIR, window_iter, np_is_ndim
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE


class TemporalMixin:
	"""
	A Mixin for models which look at the entire effective history of a data point at once.
	This is in contrast to models like RNNs where each data point may be a group of timesteps fed in sequence.
	"""

	def get_obs_shape(self, f_shape):
		"""
		Gets the shape of one observation before the final preprocessing.
		
		Args:
			f_shape	(tuple): shape of the feature data

		Returns:
			A tuple of form (channels, win_size)
		"""
		return {
			len(f_shape)==1: (1, 1),				# (num_obs,)					-> (1, 1)
			len(f_shape)==2: (f_shape[-1], 1),		# (num_obs, channels)			-> (channels, 1)
			len(f_shape)==3: f_shape[-2:]			# (num_obs, channels, win_size)	-> (channels, win_size)
		}.get(True, f_shape[-2:])

	def preproc(self, params, data):
		"""
		Reshaping transform for temporal data.

		Runs a "moving window unstack" operation through the first data such that each row of the result contains the history
		of the original up to and including that row based on a input_windows parameter in params. The input_windows
		determines how far back the history each row will record; a input_windows of '1' results in no change.
		This method also adds a singleton dimension between the first and second after the moving window unstack; this is to
		denote the "number of channels" for CNN based learning algorithms.
		
		example with input_windows of '2':
													0 | a b c 
													1 | d e f ---> 1 | a b c d e f
													2 | g h i      2 | d e f g h i
													3 | j k l      3 | g h i j k l

		All data after the first tuple item are assumed to be label/target vectors and are reshaped to align with the new first
		tuple item.

		Args:
			params (dict): params dictionary
			data (tuple): tuple of data with features as first element

		Returns:
			Tuple of reshaped data
		"""
		# Reshape features into overlapping moving window samples
		f = np.array([np.concatenate(vec, axis=-1) for vec in window_iter(data[0], n=params['input_windows'])])

		if (len(f.shape) < 3):
			f = np.expand_dims(f, 1) # Add a singleton dimension for single channel data if needed

		l = []
		for vec in data[1:]:
			r = vec[params['input_windows']-1:]								# Realign by dropping lables prior to the first step
			m = np.expand_dims(r, 1) if (np_is_ndim(vec)) else r 			# Make array vector of vectors if it is one dimensional
			l.append(m)

		return (f, *l)
