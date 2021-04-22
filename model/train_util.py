"""
Kevin Patel
"""
import sys
import os
from os import sep
from functools import partial
import math
import logging

import numpy as np
import pandas as pd
import more_itertools
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from common_util import MODEL_DIR, identity_fn, is_type, is_ser, is_valid, isnt, np_inner, get0, midx_split, pd_rows, pd_midx_to_arr, df_midx_restack, pd_to_np
from common_util import np_assert_identical_len_dim, window_iter, np_truncate_split_1d, np_truncate_vstack_2d
from model.common import PYTORCH_MODELS_DIR, PYTORCH_LOSS_CLF, PYTORCH_LOSS_REG, TEST_RATIO, VAL_RATIO


# ***** Conversion to Numpy *****
def pd_get_np_tvt(pd_obj, as_midx=True, train_ratio=.6):
	"""
	Return the train, val, test numpy splits of a pandas object

	Args:
		pd_obj (pd.DataFrame|pd.Series): data to split
		as_midx (bool): whether to interpret as MultiIndex indexed object
		train_ratio (float (0,1)): train ratio, remainder is equally split among val/test

	Returns:
		split data as numpy arrays
	"""
	raise DeprecationWarning('use pd_to_np_tvt instead')
	tv_ratio = (1-train_ratio)/2
	train_idx, val_idx, test_idx = midx_split(pd_obj.index, train_ratio, tv_ratio, tv_ratio)
	train_pd, val_pd, test_pd = pd_obj.loc[train_idx], pd_obj.loc[val_idx], pd_obj.loc[test_idx]

	if (as_midx and is_type(pd_obj.index, pd.core.index.MultiIndex)):
		train_np, val_np, test_np = map(pd_midx_to_arr, map(lambda d: d if (is_ser(d)) else d.stack(), [train_pd, val_pd, test_pd]))
	else:
		train_np, val_np, test_np = train_pd.values, val_pd.values, test_pd.values
	return train_np, val_np, test_np

def pd_to_np_tvt(pd_obj, train_ratio=7/11):
	"""
	Return the train, val, test numpy splits of a pandas object as a tuple of numpy tensors.
	Works with MultiIndex DataFrames.

	Args:
		pd_obj (pd.DataFrame|pd.Series): data to split
		train_ratio (float (0,1)): training set ratio, remainder is equally split among val/test

	Returns:
		(train, val, test) data as a tuple of numpy tensors
	"""
	tv_ratio = (1-train_ratio)/2
	train_idx, val_idx, test_idx = midx_split(pd_obj.index, train_ratio, tv_ratio, tv_ratio)
	train_df, val_df, test_df = map(partial(pd_rows, pd_obj), (train_idx, val_idx, test_idx))
	if (is_type(pd_obj.index, pd.MultiIndex)):
		train_df, val_df, test_df = map(df_midx_restack, (train_df, val_df, test_df))
	return tuple(map(pd_to_np, (train_df, val_df, test_df)))

def pd_to_np_purged_kfold(pd_obj, k=5):
	"""
	Return the purged k fold cross validation numpy splits of a pandas object as a tuple of numpy tensors.
	Works with MultiIndex DataFrames.

	Args:
		pd_obj (pd.DataFrame|pd.Series): data to split
		k (int>1): number of cross validation folds

	Returns:
		data as a tuple of numpy tensors
	"""
	raise NotImplementedError()
	#np_obj = pd_to_np(pd_obj)
	#sklearn.model_selection.TimeSeriesSplit
	#tv_ratio = (1-train_ratio)/2
	#train_idx, val_idx, test_idx = midx_split(pd_obj.index, train_ratio, tv_ratio, tv_ratio)
	#train_df, val_df, test_df = map(partial(pd_rows, pd_obj), (train_idx, val_idx, test_idx))
	#if (is_type(pd_obj.index, pd.MultiIndex)):
	#	train_df, val_df, test_df = map(df_midx_restack, (train_df, val_df, test_df))
	#return tuple(map(pd_to_np, (train_df, val_df, test_df)))


# ***** Numpy Window Preprocessing *****
def temporal_preproc_3d(data, window_size, apply_idx=[0], same_dims=True):
	"""
	Reshapes a tuple of 3+ dimensional numpy tensors by sliding windows and stacking to the last dimension.
	The last two dimensions can be flattened to retain the original number of dimensions.
	Only applies the transform to the chosen indices in the tuple.

	Changes the array shape from (N, C, H, W) to (N-window_size-1, C, H, W*window_size)

	It looks something like this (window_size = 2):
				0 | a b c
				1 | d e f ---> 1 | a b c d e f
				2 | g h i      2 | d e f g h i
				3 | j k l      3 | g h i j k l

	Args:
		data (tuple): tuple of numpy arrays
		window_size (int): desired size of window (history length)
		apply_idx (iterable): indexes to apply preprocessing to,
			all other data will be truncated so that they're all the same length
		same_dims (bool): whether to flatten the last two dimensions into one,
			setting this maintains the original number of dimensions

	Returns:
		Tuple of reshaped data
	"""
	preproc = []

	for i, d in enumerate(data):
		if (i in apply_idx):
			pp = np.array([np.stack(w, axis=-1) for w in window_iter(d, n=window_size)])
			pp = pp.reshape(*pp.shape[:-2], np.product(pp.shape[-2:])) if (same_dims) else pp
		else:
			pp = d[window_size-1:] # Realign by dropping observations prior to the first step
		preproc.append(pp)

	return tuple(preproc)

def stride_preproc_3d(data, window_size):
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

	#preproc[0] = preproc[0].reshape(*preproc[0].shape[:2], np.product(preproc[0].shape[2:])) # Flatten last two dim of features
	return tuple(preproc)

def window_shifted(data, loss, window_size, window_overlap=True, feat_dim=None):
	"""
	Return passed input data shifted into windows and processed with the other options.
	Wrapper around temporal_preproc_3d and stride_preproc_3d.

	Args:
		data (tuple): tuple of numpy arrays, features are the first element
		loss (str): name of loss function to use
		window_size (int): window size to use (this will be the last dimension of each tensor)
		window_overlap (bool): whether to use overlapping or nonoverlapping windows
		feat_dim (int): dimension of resulting feature tensor, if 'None' doesn't reshape

	Returns:
		length 3 tuple of numpy arrays, features are the first element
		if a length 1 data tuple is passed in, returns a triplicate tuple as the result
	"""
	if (window_overlap):
		win_shifted = temporal_preproc_3d(data, window_size=window_size, \
			apply_idx=[0], same_dims=True)
	else:
		win_shifted = stride_preproc_3d(data, window_size=window_size)

	if (len(win_shifted) == 1):
		f = win_shifted[0]
		if (feat_dim == 1):
			f = f.reshape(np.product(f.shape))
		elif (feat_dim == 2):
			f = f.reshape(f.shape[0], np.product(f.shape[1:]))
		return (f, f, f)

	elif (len(win_shifted) == 3):
		f, l, t = win_shifted

		if (feat_dim == 1):
			f = f.reshape(np.product(f.shape))
		elif (feat_dim == 2):
			f = f.reshape(f.shape[0], np.product(f.shape[1:]))

		if (l.shape[-1] == 2):
			l_new = np.sum(l, axis=(1, 2), keepdims=False)		# Sum label matrices to scalar values
			if (l.shape[1] > 1):
				l_new += l.shape[1]		# Shift to range [0, C-1]
			if (loss in ('bce', 'bcel') and len(l_new.shape)==1):
				l_new = np.expand_dims(l_new, axis=-1)
			l = l_new
		else:
			raise NotImplementedError("code to process label with given shape not written")

		if (t.shape[-1] == 2):
			t_new = t[t!=0.0]
			assert len(t_new)==len(t), "target with zeros removed not equal to original"
			t = t_new
		else:
			raise NotImplementedError("code to process target with given shape not written")


		return (f, l, t)


# ***** Final Processing / Batchification *****
class WindowBatchSampler(torch.utils.data.Sampler):
	"""
	Pytorch Batch Sampler for Sequential window sampling.
	Can be used with a pytorch DataLoader to sample batches from a dataset
	as sequential moving windows.

	Args:
		data_source (Dataset): dataset to sample from
		batch_size (int>=1): batch window size
		batch_step_size (int>=1): step size (distance between adjacent batches)
		batch_shuffle (bool): whether to shuffle the batch order
	"""

	def __init__(self, data_source, batch_size=128, batch_step_size=1, \
		batch_shuffle=False):
		super().__init__(data_source)
		self.data_source = data_source
		self.batch_size = batch_size
		self.batch_step_size = batch_step_size
		self.batch_shuffle = batch_shuffle

	def __iter__(self):
		iterable = more_itertools.windowed(
			range(len(self.data_source)), n=self.batch_size, step=self.batch_step_size,
				fillvalue=-1)	# fillvalue of -1 ffills the last minibatch
		if (self.batch_shuffle):
			raise NotImplementedError("Need to add ability to shuffle batches")
		return iterable

	def __len__(self):
		return math.ceil((len(self.data_source)-self.batch_size)/self.batch_step_size) + 1

def batchify(data, loss, batch_size, shuffle=False, batch_step_size=None,
	batch_shuffle=False, num_workers=0, pin_memory=False):
	"""
	Return a torch.DataLoader made from a tuple of numpy arrays.

	Args:
		data (tuple): tuple of numpy arrays, features are the first element
		loss (str): name of loss function to use
		batch_size (int): batch (or batch window) size
		shuffle (bool): whether or not to shuffle the data,
			only relevant if batch_step_size is None
		batch_step_size (int): batch window step size.
			if this is None DataLoader uses its own default sampler,
			otherwise WindowBatchSampler is used as batch_sampler
		batch_shuffle (bool): whether or not to shuffle the batch order,
			only relevant if batch_step_size is not None
		num_workers (int>=0): DataLoader option - number cpu workers to attach
		pin_memory (bool): DataLoader option - whether to pin memory to gpu

	Returns:
		torch.DataLoader
	"""
	f = torch.tensor(data[0], dtype=torch.float32, requires_grad=False)
	l = torch.tensor(data[1], dtype=torch.int64, requires_grad=False)
	t = torch.tensor(data[2], dtype=torch.float32, requires_grad=False)
	ds = TensorDataset(f, l, t)
	if (isnt(batch_step_size)):
		# Uses one of torch.utils.data.{SequentialSampler, RandomSampler}
		dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
			num_workers=num_workers, pin_memory=pin_memory)
	else:
		batch_sampler = WindowBatchSampler(ds, batch_size=batch_size, \
			batch_step_size=batch_step_size, batch_shuffle=batch_shuffle)
		dl = DataLoader(ds, batch_sampler=batch_sampler, num_workers=num_workers,
			pin_memory=pin_memory)
	return dl

def get_dataloader(data, loss, window_size, window_overlap=True, \
	feat_dim=None, batch_size=128, shuffle=False, batch_step_size=None,
	batch_shuffle=False, num_workers=0, pin_memory=False):
	"""
	Convenience function to return batches after calling window_lag on data.

	Args:
		data (tuple): tuple of numpy arrays, features are the first element
		loss (str): name of loss function to use
		window_size (int): window size to use (this will be the last dim of each tensor)
		window_overlap (bool): whether to use overlapping or nonoverlapping windows
		feat_dim (int): dimension of resulting feature tensor, if 'None' doesn't reshape
		batch_size (int): batch (or batch window) size
		shuffle (bool): whether or not to shuffle the data,
			only relevant if batch_step_size is None
		batch_step_size (int): batch window step size.
			if this is None DataLoader uses its own default sampler,
			otherwise WindowBatchSampler is used as batch_sampler
		batch_shuffle (bool): whether or not to shuffle the batch order,
			only relevant if batch_step_size is not None
		num_workers (int>0): DataLoader option - number cpu workers to attach
		pin_memory (bool): DataLoader option - whether to pin memory to gpu

	Returns:
		torch.DataLoader
	"""
	return batchify(
		data=window_shifted(data, loss, window_size, window_overlap, feat_dim),
		loss=loss, batch_size=batch_size, shuffle=shuffle,
		batch_step_size=batch_step_size, batch_shuffle=batch_shuffle,
		num_workers=num_workers, pin_memory=pin_memory)

