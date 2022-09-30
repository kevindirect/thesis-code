"""
Kevin Patel
"""
import sys
import os
from os.path import sep
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from common_util import DATA_DIR, NestedDefaultDict, load_df, isnt, is_valid, np_truncate_vstack_2d
from data.common import PROC_NAME, DATA_NAME
from data.window_util import overlap_win_preproc_3d, stride_win_preproc_3d, WindowBatchSampler
from model.metrics_util import BenchmarksMixin


class XGDataModule(BenchmarksMixin, pl.LightningDataModule):
	"""
	Experiment Group Data Module
	Defines the data pipeline from experiment group data on disk to
	pytorch dataloaders which can be used for model training.

	There are three steps to this process:
		1. prepare_data()
		2. setup()
		3. {train, val, test}_dataloader()
"""

	def __init__(self, params_t, proc_name=PROC_NAME, data_name=DATA_NAME, asset_name="SPX",
		feature_name="price,ivol", target_name="rvol_minutely_r²_sum", return_name="ret_daily_R"):
		super().__init__()
		self.params_t = params_t
		self.proc_name = proc_name
		self.data_name = data_name
		self.asset_name = asset_name
		self.feature_name = feature_name
		self.target_name = target_name
		self.return_name = return_name
		self.name = f"{asset_name}{sep}{target_name}{sep}{feature_name}"
		self.ddir = f"{DATA_DIR}{self.proc_name}{sep}{self.data_name}{sep}{self.asset_name}"
		self.target_names = None
		self.fshape = None
		if (self.data_name == "frd"):
			self.day_size = 391 # number of data points per trading day

	def prepare_feature(self, split="train"):
		"""
		returns data shaped like (n, C, H, W).
			n - observation index
			C - channels (financial instrument)
			H - columns (open, high, low, close, etc)
			W - window (time sequence)
		"""
		dfs = [load_df(name, f"{self.ddir}/{split}/feature").set_index("datetime")
			for name in self.feature_name.split(',')]
		assert all((df.index == dfs[0].index).all() for df in dfs)
		arrs = [np_truncate_vstack_2d(df.to_numpy(), self.day_size) for df in dfs]
		assert all(arr.shape == arrs[0].shape for arr in arrs)
		return np.stack(arrs, axis=1)

	def prepare_target(self, split="train"):
		price = load_df("price", f"{self.ddir}/{split}/target").set_index("datetime")
		np_index = price.index.to_numpy()
		np_return = price.loc[:, self.return_name].to_numpy()
		np_target = price.loc[:, self.target_name].to_numpy()
		assert np_index.shape == np_return.shape == np_target.shape
		return np_index, np_return, np_target

	def prepare_data(self):
		"""
		Load the desired dataframes/splits, convert to numpy,
		and reshape the features if needed.
		Does not depend on params_t.
		"""
		self.data = NestedDefaultDict()
		for split in ["train", "val", "test"]:
			np_feature = self.prepare_feature(split)
			np_index, np_return, np_target = self.prepare_target(split)
			assert np_feature.shape[0] == np_index.shape[0]
			self.data[[split, "feature"]] = np_feature
			self.data[[split, "index"]] = np_index
			self.data[[split, "return"]] = np_return
			self.data[[split, "target"]] = np_target

	def setup(self, stage=None):
		"""
		Apply moving window to the features,
		make the TensorDatasets and samplers used to later create DataLoaders.
		Depends on params_t.
		"""
		self.dataset, self.index = {}, {}
		for split in ["train", "val", "test"]:
			shifted = XGDataModule.window_shift((
				self.data[[split, "feature"]],
				self.data[[split, "target"]],
				self.data[[split, "return"]],
				self.data[[split, "index"]]
			))
			self.dataset[split] = XGDataModule.get_dataset(shifted[:-1])
			self.index[split] = shifted[-1]

		self.sampler, self.interval = None, None
		if (is_valid(self.params_t['batch_step_size'])):
			self.sampler = {
				split: WindowBatchSampler(ds,
					batch_size=self.params_t['batch_size'],
					batch_step_size=self.params_t['batch_step_size'],
					method='trunc',
					batch_shuffle=self.is_shuffle(split))
				for split, ds in self.dataset.items()
			}

			self.interval = NestedDefaultDict()
			for split, samp in self.sampler.items():
				interval_c = (0, samp.get_step_end())
				interval_t = (self.params_t['batch_step_size'], samp.get_data_end())
				self.interval[[split, "context"]] = interval_c
				self.interval[[split, "target"]] = interval_t

			# first to last target set indices:
			# self.idx = {split: idx[-1][self.start[split]:self.end[split]]
			# 	for split, idx in self.index.items()}

	def get_fshape(self):
		"""
		Shape of each feature observation.
		Depends on params_t.

		If feature tensor is shaped (n, C, H, W) this will be (C, H, W), where:
			* C: channel (matrix of data)
			* H: data column (series)
			* W: data row (time dimension)
		"""
		if (isnt(self.fshape)):
			_fshape = list(self.data[["train", "feature"]][0].shape)
			_fshape[-1] *= self.params_t["window_size"]
			self.fshape = tuple(_fshape)
		return self.fshape

	def update(self, new):
		if (self.params_t != new):
			self.params_t = new
			self.fshape = None
			self.setup()

	def get_dataloader(self, split):
		"""
		Return a torch.DataLoader

		Args:
			split (str):

		Returns:
			torch.DataLoader
		"""
		if (is_valid(self.sampler and self.sampler[split])):
			dl = DataLoader(self.dataset[split],
				batch_sampler=self.sampler[split],
				num_workers=self.params_t['num_workers'],
				pin_memory=self.params_t['pin_memory'])
		else:
			# Uses one of torch.utils.data.{SequentialSampler, RandomSampler}
			dl = DataLoader(self.dataset[split], batch_size=self.params_t['batch_size'],
				shuffle=self.is_shuffle(split), drop_last=False,
				num_workers=self.params_t['num_workers'],
				pin_memory=self.params_t['pin_memory'])
		return dl

	is_shuffle = lambda self, split: self.params_t['train_shuffle'] and split=='train'
	train_dataloader = lambda self: self.get_dataloader('train')
	val_dataloader = lambda self: self.get_dataloader('val')
	test_dataloader = lambda self: self.get_dataloader('test')

	def get_target_names(self, split="train"):
		if (isnt(self.target_names)):
			price = load_df("price", f"{self.ddir}/{split}/target").set_index("datetime")
			self.target_names = list(price.columns)
		return self.target_names

	@staticmethod
	def rename_ohlc(df, pfx):
		newcols = {
			"open": f"{pfx}_open",
			"high": f"{pfx}_high",
			"low": f"{pfx}_low",
			"close": f"{pfx}_close"
		}
		return df.rename(columns=newcols)

	@staticmethod
	def window_shift(data, window_size=1, window_overlap=True):
		"""
		Return passed input data reshaped into moving windows by
		reducing the first dimension and expanding the last dimension.
		Wrapper around temporal_preproc_3d and stride_preproc_3d.

		The last dimension of the features will be multiplied in length
		by the window_size argument, and this number of elements will be
		truncated off the first dimension. Thus if a window size of 1 is passed
		in, no shifting occurs. Non feature arrays are truncated so they
		still line up with the features.

		Args:
			data (tuple): tuple of numpy arrays, features are the first element
			window_size (int): window size to use (will be last dimension of each tensor)
			window_overlap (bool): whether to use overlapping or nonoverlapping windows

		Returns:
			tuple of numpy arrays
		"""
		assert window_size >= 1
		if (window_overlap):
			shifted = overlap_win_preproc_3d(data, window_size=window_size, \
				same_dims=True)
		else:
			shifted = stride_win_preproc_3d(data, window_size=window_size)
		return shifted

	@staticmethod
	def get_dataset(data):
		"""
		Return TensorDataset of (features, targets, returns)

		Args:
			data (tuple): tuple of numpy arrays, features are the first element

		Returns:
			torch.TensorDataset
		"""
		f = torch.tensor(data[0], dtype=torch.float32, requires_grad=False)
		t = torch.tensor(data[1], dtype=torch.float32, requires_grad=False)
		r = torch.tensor(data[2], dtype=torch.float32, requires_grad=False)
		return TensorDataset(f, t, r)

