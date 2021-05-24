"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from common_util import MODEL_DIR, DT_FMT_YMD, pd_split_ternary_to_binary, rectify_json, pairwise, df_del_midx_level, df_randlike, is_valid
from model.common import ASSETS, INTRADAY_LEN, INTERVAL_YEARS
from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs, get_hardcoded_feature_dfs, get_hardcoded_label_target_dfs, dfs_get_common_interval_data
from model.train_util import pd_to_np_tvt, pd_tvt_idx_split, get_dataloader
from model.metrics_util import BuyAndHoldStrategy, OptimalStrategy


class XGDataModule(pl.LightningDataModule):
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
		t_params (dick): training hyperparameters
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
		self.name = (f'{self.interval[0]}_{self.interval[1]}'
			f'_{self.ldata_name}_{self.fdata_name}').replace(',', '_')

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

		self.data = dfs_get_common_interval_data((fdata, ldata, tdata),
			interval=self.interval)

	def setup(self):
		"""
		Split the data into {train, val, test} splits and convert to numpy arrays.
		"""
		self.train, self.val, self.test = zip(*map(pd_to_np_tvt, self.data))
		self.train_idx, self.val_idx, self.test_idx = pd_tvt_idx_split(self.data[1])
		self.fobs = self.get_fobs()

		for split in (self.train, self.val, self.test):
			assert all(len(d)==len(split[0]) for d in split), \
				"length of data within each split must be identical"

		for train, val, test in zip(self.train, self.val, self.test):
			assert train.shape[1:] == val.shape[1:] == test.shape[1:], \
				"shape of f, l, t data must be identical across splits"

	def get_fobs(self):
		fobs = list(self.train[0].shape[1:])
		fobs[-1] = fobs[-1] * self.t_params['window_size']
		fobs = tuple(fobs)
		return fobs

	def update_params(self, new):
		old_window_size = self.t_params['window_size']
		self.t_params = new
		if (self.t_params['window_size'] != old_window_size):
			self.fobs = self.get_fobs()

	train_dataloader = lambda self: get_dataloader(
		data=self.train,
		loss=self.t_params['loss'],
		window_size=self.t_params['window_size'],
		window_overlap=True,
		feat_dim=self.t_params['feat_dim'],
		batch_size=self.t_params['batch_size'],
		batch_step_size=self.t_params['batch_step_size'],
		batch_shuffle=self.t_params['train_shuffle'],
		num_workers=self.t_params['num_workers'],
		pin_memory=self.t_params['pin_memory'])

	val_dataloader = lambda self: get_dataloader(
		data=self.val,
		loss=self.t_params['loss'],
		window_size=self.t_params['window_size'],
		window_overlap=True,
		feat_dim=self.t_params['feat_dim'],
		batch_size=self.t_params['batch_size'],
		batch_step_size=self.t_params['batch_step_size'],
		num_workers=self.t_params['num_workers'],
		pin_memory=self.t_params['pin_memory'])

	test_dataloader = lambda self: get_dataloader(
		data=self.test,
		loss=self.t_params['loss'],
		window_size=self.t_params['window_size'],
		window_overlap=True,
		feat_dim=self.t_params['feat_dim'],
		batch_size=self.t_params['batch_size'],
		batch_step_size=self.t_params['batch_step_size'],
		num_workers=self.t_params['num_workers'],
		pin_memory=self.t_params['pin_memory'])

	def get_benchmarks(self, train_len=None, val_len=None, test_len=None):
		"""
		Return benchmarks calculated from loaded data as a JSON serializable dict.
		"""
		bench_dict = {}
		bench_stats = (
			lambda d, pfx: {
				f'{pfx}_label_dist': pd.Series(d, dtype=int) \
					.value_counts(normalize=True).to_dict()
			},
		)
		bench_strats = (
			BuyAndHoldStrategy(compounded=False),
			# BuyAndHoldStrategy(compounded=True),
			OptimalStrategy(compounded=False),
			# OptimalStrategy(compounded=True)
		)

		for pfx, flt, flt_idx, split_len in zip( \
			('train', 'val', 'test'), \
			(self.train, self.val, self.test), \
			(self.train_idx, self.val_idx, self.test_idx), \
			(train_len, val_len, test_len)):
			l = np.sum(flt[1], axis=(1, 2), keepdims=False)
			t = flt[2]
			i = flt_idx.droplevel(1)
			if (is_valid(split_len)):
				l = l[:split_len]
				t = t[:split_len]
				i = i[:split_len]
			t = torch.tensor(t[t!=0], dtype=torch.float32, requires_grad=False)
			# t = np.sum(flt[2], axis=(1, 2), keepdims=False)

			bench_dict[f'{pfx}_date_range_start'] = i[0].strftime(DT_FMT_YMD)
			bench_dict[f'{pfx}_date_range_end'] = i[-1].strftime(DT_FMT_YMD)
			bench_dict[f'{pfx}_date_range_len'] = len(i)
			for stat in bench_stats:
				bench_dict.update(stat(l, pfx))

			for strat in bench_strats:
				strat.update(t)
				vals = strat.compute(pfx=pfx)
				bench_dict.update(vals)
				strat.reset()
		return rectify_json(bench_dict)

	def get_benchmark_strats(self, train_len=None, val_len=None, test_len=None):
		"""
		Return benchmarks objects.
		"""
		bench_strats = {pfx: {} for pfx in ('train', 'val', 'test')}

		for pfx, flt, flt_idx, split_len in zip( \
			('train', 'val', 'test'), \
			(self.train, self.val, self.test), \
			(self.train_idx, self.val_idx, self.test_idx), \
			(train_len, val_len, test_len)):
			strats = (
				BuyAndHoldStrategy(compounded=False),
				# BuyAndHoldStrategy(compounded=True),
				OptimalStrategy(compounded=False),
				# OptimalStrategy(compounded=True)
			)
			# t = np.sum(flt[2], axis=(1, 2), keepdims=False)
			t = flt[2]
			i = flt_idx.droplevel(1)
			if (is_valid(split_len)):
				t = t[:split_len]
				i = i[:split_len]
			t = torch.tensor(t[t!=0], dtype=torch.float32, requires_grad=False)
			for strat in strats:
				strat.update(t)
				bench_strats[pfx][strat.name] = strat
			bench_strats[pfx]['index'] = i
		return bench_strats

