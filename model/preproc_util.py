"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from common_util import np_assert_identical_len_dim, window_iter
from model.common import PYTORCH_ACT_MAPPING


# def temporal_preproc(data, window_size):
# 	"""
# 	Reshaping transform for temporal data. All data must have same number of dimensions and observations.

# 	Runs a "moving window unstack" operation through the first data such that each row of the result contains the history
# 	of the original up to and including that row based on window_size. The window_size
# 	determines how far back the history each row will record; a window_size of '1' results in no change.
# 	This method also adds a singleton dimension between the first and second after the moving window unstack; this is to
# 	denote the "number of channels" for CNN based learning algorithms.

# 	example with window_size of '2':
# 				0 | a b c
# 				1 | d e f ---> 1 | a b c d e f
# 				2 | g h i      2 | d e f g h i
# 				3 | j k l      3 | g h i j k l

# 	All data after the first tuple item are assumed to be label/target vectors and are reshaped to align with the new first
# 	tuple item.

# 	Args:
# 		data (tuple): tuple of numpy data with features as first element
# 		window_size (int): desired size of window (history length)

# 	Returns:
# 		Tuple of reshaped data
# 	"""
# 	np_assert_identical_len_dim(*data)
# 	# Reshape features into overlapping moving window samples
# 	f = np.array([np.concatenate(vec, axis=-1) for vec in window_iter(data[0], n=window_size)])
# 	rest = [vec[window_size-1:] for vec in data[1:]]  # Realign by dropping observations prior to the first step
# 	np_assert_identical_len_dim(f, *rest)
# 	return (f, *rest)


def temporal_preproc(data, window_size, apply_idx=[0]):
	"""
	Reshaping transform for temporal data. All data must have same number of dimensions and observations.

	Runs a "moving window unstack" operation through the first data such that each row of the result contains the history
	of the original up to and including that row based on window_size. The window_size
	determines how far back the history each row will record; a window_size of '1' results in no change.
	This method also adds a singleton dimension between the first and second after the moving window unstack; this is to
	denote the "number of channels" for CNN based learning algorithms.

	example with window_size of '2':
				0 | a b c
				1 | d e f ---> 1 | a b c d e f
				2 | g h i      2 | d e f g h i
				3 | j k l      3 | g h i j k l

	All data after the first tuple item are assumed to be label/target vectors and are reshaped to align with the new first
	tuple item.

	Args:
		data (tuple): tuple of numpy data with features as first element
		window_size (int): desired size of window (history length)
		apply_idx (iterable): indexes to apply preprocessing to, all other data will be truncated to match

	Returns:
		Tuple of reshaped data
	"""
	np_assert_identical_len_dim(*data)
	preproc = []
	for i, d in enumerate(data):
		if (i in apply_idx):
			# Reshape features into overlapping moving window samples
			preproc.append(np.array([np.concatenate(vec, axis=-1) for vec in window_iter(d, n=window_size)]))
		else:
			preproc.append(d[window_size-1:]) # Realign by dropping observations prior to the first step
	np_assert_identical_len_dim(*preproc)
	return tuple(preproc)

