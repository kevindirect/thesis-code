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

from common_util import MODEL_DIR, pd_split_ternary_to_binary, pairwise, df_del_midx_level, df_randlike, is_valid
from model.common import ASSETS, INTRADAY_LEN, INTERVAL_YEARS
from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs, get_hardcoded_feature_dfs, get_hardcoded_label_target_dfs, dfs_get_common_interval_data
from model.train_util import pd_to_np_tvt, pd_tvt_idx_split, window_shifted, WindowBatchSampler, get_dataset
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

	Can be used directly with pytorch-lightning Trainer.fit().

	Args:
		t_params (dict): training hyperparameters
		asset_name (str): asset name
		fdata_name (d_rand|d_pba|d_vol|d_buzz|d_nonbuzz|h_rand|h_pba|h_vol|h_buzz):
			feature data name
		ldata_name (dcur|ddir|ddir1|ddir2|dxfbdir1|dxfbdir2): label/target data name
		interval (tuple): tuple of years to index over and intersect feature, label,
			and target data over
		overwrite_cache (bool): overwrite the cache when loading xg data from disk
	"""

	def __init__(self, t_params, asset_name, fdata_name, ldata_name, fret=None,
		interval=INTERVAL_YEARS, overwrite_cache=False):
		super().__init__()
		self.t_params = t_params
		self.asset_name = asset_name
		self.fdata_name = fdata_name
		self.ldata_name = ldata_name
		self.fret = fret
		self.interval = interval
		self.overwrite_cache = overwrite_cache
		self.name = XGDataModule.get_name(self.interval, self.fdata_name, self.ldata_name)

	def prepare_data(self):
		"""
		Read xg data from disk, choose the desired dataframes, restructure them,
		and index/intersect the desired time series interval
		"""
		fd = get_xg_feature_dfs(self.asset_name, overwrite_cache=self.overwrite_cache)

		if (self.fdata_name.startswith('h_rand')):
			rl = get_hardcoded_feature_dfs(fd, 'h_pba_h', cat=True,
				ret=None, pfx='0')
			fdata = df_randlike(rl, cols=INTRADAY_LEN)
		else:
			fds = [get_hardcoded_feature_dfs(fd, fdata_name, cat=True,
				ret=self.fret, pfx=i)
				for i, fdata_name in enumerate(self.fdata_name.split(','))]
			fdata = pd.concat(dfs_get_common_interval_data(fds), axis=0)

		if (self.ldata_name == 'dcur'):
			# Sanity check: 'Predict' the present ddir(t-1)
			ldata = pd_split_ternary_to_binary(df_del_midx_level( \
				fd['d']['pba']['ddir']['pba_hoc_hdxret_ddir'] \
				.rename(columns={-1:'pba_hoc_hdxret_ddir'}), loc=1) \
				.replace(to_replace=-1, value=0).astype(int))
			tdata = pd_split_ternary_to_binary(df_del_midx_level( \
				fd['d']['pba']['dret']['pba_hoc_hdxret_dret'] \
				.rename(columns={-1:'pba_hoc_hdxret_dret'}), loc=1))
		else:
			ld, td = get_xg_label_target_dfs(self.asset_name,
				overwrite_cache=self.overwrite_cache)
			ldata, tdata = get_hardcoded_label_target_dfs(ld, td, self.ldata_name)

		self.raw_df = dfs_get_common_interval_data((fdata, ldata, tdata),
			interval=self.interval)

	def setup(self):
		"""
		Split the data into {train, val, test} splits and convert to numpy arrays,
		apply window_shifted per split to get win_{split} arrays,
		and make the TensorDatasets and samplers used to later create DataLoaders.
		"""
		raw_train, raw_val, raw_test = zip(*map(pd_to_np_tvt, self.raw_df))
		idx_train, idx_val, idx_test = zip(*map(pd_tvt_idx_split, self.raw_df))

		for split in (raw_train, raw_val, raw_test):
			assert all(len(d)==len(split[0]) for d in split), \
				"length of data within each split must be identical"
		for train, val, test in zip(raw_train, raw_val, raw_test):
			assert train.shape[1:] == val.shape[1:] == test.shape[1:], \
				"shape of f, l, t data must be identical across splits"

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

	@staticmethod
	def get_name(interval, fdata_name, ldata_name):
		return (f'{interval[0]}_{interval[1]}'
			f'_{ldata_name}_{fdata_name}').replace(',', '_')

