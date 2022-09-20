"""
Kevin Patel
"""
import sys
import os
import logging
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from common_util import MODEL_DIR, pd_split_ternary_to_binary, pairwise, df_randlike, is_valid
from model.common import ASSETS
# from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs, get_hardcoded_feature_dfs, get_hardcoded_label_target_dfs, dfs_get_common_interval_data
# from model.train_util import pd_to_np_tvt, pd_tvt_idx_split, window_shifted, WindowBatchSampler, get_dataset
from model.metrics_util import BenchmarksMixin


class XGDataModule(BenchmarksMixin, pl.LightningDataModule):
	"""
	Experiment Group Data Module
	Defines the data pipeline from experiment group data on disk to
	pytorch dataloaders which can be used for model training.

	There are three steps to this process:
		1. prepare_data: read the desired xg dataframes data from disk,
			restructure them, and index/intersect over the desired time interval
		2. setup: split the data into {train, val, test} and convert to numpy
		3. {train, val, test}_dataloader: do some final processing on the numpy
			arrays and return the data as a DataLoader of pytorch tensors
	"""

	def __init__(self, t_params, stage_name="001", data_name="frd", asset_name="SPX",
		target_name="minutely_r²_sum"):
		super().__init__()
		self.t_params = t_params
		self.stage_name = stage_name
		self.data_name = data_name
		self.asset_name = asset_name
		self.target_name = target_name
		self.overwrite_cache = overwrite_cache
		self.name = f"{stage_name}_{data_name}_{asset_name}_{target_name}"

	def get_raw_dfs(self):
		ddir = f"{DATA_DIR}{self.stage_name}/{self.data_name}"
		price = load_df(self.asset_name, f"{ddir}/price/features").set_index("datetime")
		iv = load_df(self.asset_name, f"{ddir}/iv/features").set_index("datetime")
		target = load_df(self.asset_name, f"{ddir}/price/targets").set_index("datetime")
		return price, iv, target

	def prepare_data(self):
		"""
		Read xg data from disk, choose the desired dataframes, restructure them,
		and index/intersect the desired time series interval
		"""
		self.price, self.iv, self.target = self.get_raw_dfs()
		# self.raw_df = dfs_get_common_interval_data((fdata, ldata, tdata),
		# 	interval=self.interval)

	def setup(self, stage=None):
		"""
		Split the data into {train, val, test} splits and convert to numpy arrays,
		apply window_shifted per split to get win_{split} arrays,
		and make the TensorDatasets and samplers used to later create DataLoaders.
		"""
		raw_train, raw_val, raw_test = zip(*map(pd_to_np_tvt, self.raw_df))
		idx_train, idx_val, idx_test = zip(*map(pd_tvt_idx_split, self.raw_df))

		# for split in (raw_train, raw_val, raw_test):
		# 	assert all(len(d)==len(split[0]) for d in split), \
		# 		"length of data within each split must be identical"
		# for train, val, test in zip(raw_train, raw_val, raw_test):
		# 	assert train.shape[1:] == val.shape[1:] == test.shape[1:], \
		# 		"shape of f, l, t data must be identical across splits"

		self.raw_idx = {
			'train': idx_train,
			'val': idx_val,
			'test': idx_test
		}
		self.raw = {
			'train': raw_train,
			'val': raw_val,
			'test': raw_test
		}
		self.fobs = self.get_fobs()

		win_fn = partial(window_shifted,
			loss=self.t_params['loss'],
			window_size=self.t_params['window_size'],
			window_overlap=True,
			feat_dim=self.t_params['feat_dim'])

		self.win = {split: win_fn(d) for split, d in self.raw.items()}
		self.ds = {split: get_dataset(d) for split, d in self.win.items()}
		if (is_valid(self.t_params['batch_step_size'])):
			self.smp = {
				split: WindowBatchSampler(ds,
					batch_size=self.t_params['batch_size'],
					batch_step_size=self.t_params['batch_step_size'],
					method='trunc',
					batch_shuffle=self.is_shuffle(split))
				for split, ds in self.ds.items()
			}
			self.starts = {
				split: (0, self.t_params['batch_step_size'])
					for split, smp in self.smp.items()
			}
			self.ends = {
				split: (smp.get_step_end(), smp.get_data_end())
					for split, smp in self.smp.items()
			}

			# first to last target set indices:
			self.start = {split: idxs[1] for split, idxs in self.starts.items()}
			self.end = {split: idxs[1] for split, idxs in self.ends.items()}
			self.idx = {split: idx[-1].droplevel(1)[self.start[split]:self.end[split]]
				for split, idx in self.raw_idx.items()}
		else:
			self.smp = None
			self.starts = None
			self.ends = None
			self.start = None
			self.end = None
			self.idx = None

	def get_fobs(self):
		"""
		For a time series of 24 hour days, this would be 24.
		For 8 hour days, 8.
		And so on.
		"""
		fobs = list(self.raw['train'][0].shape[1:])
		fobs[-1] = fobs[-1] * self.t_params['window_size']
		fobs = tuple(fobs)
		return fobs

	def update_params(self, new):
		if (self.t_params != new):
			self.t_params = new
			self.setup()

	def get_dataloader(self, split):
		"""
		Return a torch.DataLoader

		Args:
			split (str):

		Returns:
			torch.DataLoader
		"""
		dataset = self.ds[split]
		sampler = self.smp and self.smp[split]

		if (is_valid(sampler)):
			dl = DataLoader(dataset, batch_sampler=sampler,
				num_workers=self.t_params['num_workers'],
				pin_memory=self.t_params['pin_memory'])
		else:
			# Uses one of torch.utils.data.{SequentialSampler, RandomSampler}
			dl = DataLoader(dataset, batch_size=self.t_params['batch_size'],
				shuffle=self.is_shuffle(split), drop_last=False,
				num_workers=self.t_params['num_workers'],
				pin_memory=self.t_params['pin_memory'])
		return dl

	is_shuffle = lambda self, split: self.t_params['train_shuffle'] and split=='train'
	train_dataloader = lambda self: self.get_dataloader('train')
	val_dataloader = lambda self: self.get_dataloader('val')
	test_dataloader = lambda self: self.get_dataloader('test')

