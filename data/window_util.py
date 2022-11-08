"""
Kevin Patel
"""
import sys
import os
import math
import logging

import numpy as np
import pandas as pd
import more_itertools
import torch

from common_util import window_iter, trunc_step_window_iter, pt_random_choice
from data.common import dum


def overlap_win_preproc_3d(data, window_size, same_dims=True):
	"""
	Reshapes a tuple of 3+ dimensional numpy tensors by sliding windows and stacking
	to the last dimension.

	The last two dimensions can be flattened to retain the original number of dimensions.
	Only applies the transform to the chosen indices in the tuple.

	Any vectors passed into the tuple are truncated to maintain alignment.

	Changes the array shape from (N, C, H, W) to (N-window_size-1, C, H, W*window_size)

	It looks something like this (window_size = 2):
				0 | a b c
				1 | d e f ---> 1 | a b c d e f
				2 | g h i      2 | d e f g h i
				3 | j k l      3 | g h i j k l

	Args:
		data (tuple): tuple of numpy arrays
		window_size (int): desired size of window (history length)
		same_dims (bool): whether to flatten the last two dimensions into one,
			setting this maintains the original number of dimensions

	Returns:
		Tuple of reshaped data
	"""
	preproc = []

	for i, d in enumerate(data):
		if (d.ndim > 1):
			pp = np.array([np.stack(w, axis=-1) for w in window_iter(d, n=window_size)])
			pp = pp.reshape(*pp.shape[:-2], np.product(pp.shape[-2:])) if (same_dims) else pp
		else:
			pp = d[window_size-1:] # Realign by dropping observations prior to the first step
		preproc.append(pp)
	# assert all(d.shape[0]==preproc[0].shape[0] for d in preproc[1:])
	return tuple(preproc)

def stride_win_preproc_3d(data, window_size):
	"""
	Reshape with non-overlapping windows.
	Slices a multi dimensional array at the first dimension using a window size
	and stacks them horizontally so that each slice becomes one row/observation.

	Changes the array shape from (N, C, H, W) to (N//window_size, C, H, W*window_size)

	It looks something like this (window_size = 2):
				0 | a b c
				1 | d e f ---> 1 | a b c d e f
				2 | g h i
				3 | j k l      2 | g h i j k l

	Args:
		data (tuple): tuple of numpy arrays
		window_size (int): desired size of window size and stride

	Returns:
		Tuple of reshaped data
	"""
	preproc = []

	for i, d in enumerate(data):
		sub_arr = [np.swapaxes(d[i:i+window_size], 0, 1) \
			for i in range(0, len(d), window_size)]
		trunc_arr = sub_arr[:-1] if (sub_arr[0].shape != sub_arr[-1].shape) else sub_arr
		preproc.append(np.stack(trunc_arr, axis=0))

	return tuple(preproc)

def get_np_collate_fn(context_size, target_size, overlap_size=0, resample_context=False):
	def split_ct(x):
		"""
		Split into context and target sets
		"""
		contexts, targets = [], []
		for obs in trunc_step_window_iter(x, n=context_size+target_size, step=context_size):
			cpoints = obs[:context_size]
			if (resample_context):
				cpoints = pt_random_choice(cpoints, replacement=True)
			contexts.append(cpoints)
			targets.append(obs[context_size-overlap_size:])
		return torch.stack(contexts), torch.stack(targets)

	def np_collate_fn(batch):
		"""
		Neural Process collate
		Reshapes each series from (n, ...) to (n', o, ...), where
			n is the batch size
			n' is the new batch size
			o is the observation set size (context/target size)
		"""
		i, x, y, z = map(torch.stack, zip(*batch))
		bc, bt = split_ct(torch.arange(len(i)))
		assert len(bc)==len(bt)
		return i[bc], x[bc], y[bc], z[bc], i[bt], x[bt], y[bt], z[bt]

	return np_collate_fn

class WindowBatchSampler(torch.utils.data.Sampler):
	"""
	Pytorch Batch Sampler for Sequential window sampling.
	Can be used with a pytorch DataLoader to sample batches from a dataset
	as sequential moving windows.

	Args:
		data_source (Dataset): dataset to sample from
		batch_size (int>=1): batch window size
		batch_step_size (None or int>=1): step size (distance between adjacent batches)
		batch_shuffle (bool): whether to shuffle the batch order
		method: how to deal with remainder
	"""

	def __init__(self, data_source, batch_size=128, batch_step_size=None, \
		method='trunc', batch_shuffle=False):
		super().__init__(data_source)
		self.data_source = data_source
		self.batch_size = batch_size
		self.batch_step_size = batch_step_size or batch_size
		self.batch_shuffle = batch_shuffle
		self.method = method
		assert len(self.data_source) >= self.batch_size, "must have at least one batch"

	def __iter__(self):
		if (self.method == 'ffill'):
			iterable = more_itertools.windowed(range(len(self.data_source)), \
					n=self.batch_size, step=self.batch_step_size, fillvalue=-1)
		elif (self.method == 'nfill'):
			iterable = more_itertools.windowed(range(len(self.data_source)), \
				n=self.batch_size, step=self.batch_step_size, fillvalue=None)
		elif (self.method == 'trunc'):
			iterable = more_itertools.windowed(range(self.get_data_end()), \
				n=self.batch_size, step=self.batch_step_size, fillvalue=None)

		if (self.batch_shuffle):
			raise NotImplementedError("Need to add ability to shuffle batches")

		return iterable

	def get_num_steps(self):
		"""
		Get the number of steps the iterator moves through.
		If this is zero, the iterator only goes through one batch.
		"""
		num_steps = (len(self.data_source) - self.batch_size) / self.batch_step_size
		num_steps = math.floor(num_steps) if (self.method in ('trunc',)) \
			else math.ceil(num_steps)
		return num_steps

	"""
	Get the number of windows (batches) the iterator moves through.
	"""
	__len__ = get_num_windows = lambda self: self.get_num_steps() + 1
		

	"""
	Get the first index of the last batch (ie beginning of the last step)
	"""
	get_step_end = lambda self: self.batch_step_size * self.get_num_steps()

	"""
	Get the last index from the last batch (ie end of the last step)
	"""
	get_data_end = lambda self: self.batch_size + self.get_step_end()

