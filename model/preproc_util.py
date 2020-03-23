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

from common_util import np_assert_identical_len_dim, window_iter, np_truncate_split_1d, np_truncate_vstack_2d
from model.common import PYTORCH_ACT_MAPPING


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
	raise DeprecationWarning('use temporal_preproc_3d instead')
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

def stride_preproc(data, window_size):
	"""
	Reshape by striding through data by window_size, basically a moving window with no overlapping windows.

	Args:
		data (tuple): tuple of numpy data
		window_size (int): desired size of window size and stride

	Returns:
		Tuple of reshaped data
	"""
	raise DeprecationWarning('use stride_preproc_3d instead')
	np_assert_identical_len_dim(*data)
	preproc = []
	for i, d in enumerate(data):
		d = np.squeeze(d)
		if (d.ndim == 1):
			preproc.append(np.expand_dims(np_truncate_split_1d(d, window_size), -1))
		elif (d.ndim == 2):
			preproc.append(np_truncate_vstack_2d(d, window_size))
		else:
			raise ValueError("data.ndim must be 1 or 2 after squeezing")
	np_assert_identical_len_dim(*preproc)
	return tuple(preproc)


##
# Must result in shape (number observations, channels, window size, hours/window observations) -> (n, C, W, H)

def temporal_preproc_3d(data, window_size, apply_idx=[0], flatten=True):
	"""
	Reshapes a tuple of three dimensional numpy tensors.
	Slides a window down the tensor and stacks sub-tensors to the second dimension.

	Given a tensor (N, C, H):
		(N, C, H) -> (N-window_size-1, C, window_size, H)

	Args:
		data (tuple): tuple of numpy data with features as first element
		window_size (int): desired size of window (history length)
		apply_idx (iterable): indexes to apply preprocessing to, all other data will be truncated to match
		flatten (bool): whether or not to flatten last two dimensions into one

	Returns:
		Tuple of reshaped data
	"""
	np_assert_identical_len_dim(*data)
	preproc = []
	for i, d in enumerate(data):
		if (i in apply_idx):
			pp = np.array([np.stack(vec, axis=1) for vec in window_iter(d, n=window_size)]) # Reshape features into overlapping moving window samples
			pp = pp.reshape(*pp.shape[:2], np.product(pp.shape[2:])) if (flatten) else pp	# Flatten last two dim (window and window_obs) into one
		else:
			pp = d[window_size-1:] # Realign by dropping observations prior to the first step
		preproc.append(pp)

	return tuple(preproc)


def stride_preproc_3d(data, window_size):
	"""
	Reshape by striding through data by window_size, like a moving window with non-overlapping windows.

	Given a tensor (N, C, H):
		(N, C, H) -> (N//window_size, C, window_size, H)

	Args:
		data (tuple): tuple of numpy data
		window_size (int): desired size of window size and stride

	Returns:
		Tuple of reshaped data
	"""
	np_assert_identical_len_dim(*data)
	preproc = []
	for i, d in enumerate(data):
		sub_arr = [np.swapaxes(d[i:i+window_size], 0, 1) for i in range(0, len(d), window_size)]
		trunc_arr = sub_arr[:-1] if (sub_arr[0].shape != sub_arr[-1].shape) else sub_arr
		preproc.append(np.stack(trunc_arr, axis=0))
	np_assert_identical_len_dim(*preproc)

	#preproc[0] = preproc[0].reshape(*preproc[0].shape[:2], np.product(preproc[0].shape[2:])) # Flatten last two dim of features
	return tuple(preproc)

