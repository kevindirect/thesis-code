"""
"""
import sys
import os
import logging

import pandas as pd
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from common_util import MODEL_DIR, pd_split_ternary_to_binary, rectify_json, pairwise
from model.common import ASSETS, INTRADAY_LEN, INTERVAL_YEARS
from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs, get_hardcoded_feature_dfs, get_hardcoded_label_target_dfs, get_common_interval_data
from model.train_util import pd_to_np_tvt, get_dataloader
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

	def __init__(self, t_params, asset_name, fdata_name, ldata_name,
		interval=INTERVAL_YEARS, overwrite_cache=False):
		super().__init__()
		self.t_params = t_params
		self.asset_name = asset_name
		self.fdata_name = fdata_name
		self.ldata_name = ldata_name
		self.interval = interval
		self.overwrite_cache = overwrite_cache
		self.name = (f'{self.interval[0]}_{self.interval[1]}'
			f'_{self.ldata_name}_{self.fdata_name}')

	def prepare_data(self):
		"""
		Read xg data from disk, choose the desired dataframes, restructure them,
		and index/intersect the desired time series interval
		"""
		if (self.fdata_name in ('d_rand', 'h_rand')):
			fd = None
			if (fdata_k[0] == 'd'):
				raise NotImplementedError()
			elif (fdata_k[0] == 'h'):
				raise NotImplementedError()
		else:
			fd = get_xg_feature_dfs(self.asset_name,
				overwrite_cache=self.overwrite_cache)
			fdata = get_hardcoded_feature_dfs(fd, self.fdata_name)

		if (self.ldata_name == 'dcur'):
			# Sanity check: 'Predict' the present ddir(t-1)
			fd = fd or get_xg_feature_dfs(self.asset_name)
			ldata = pd_split_ternary_to_binary(df_del_midx_level(\
				fd['d']['pba']['ddir']['pba_hoc_hdxret_ddir'] \
				.rename(columns={-1:'pba_hoc_hdxret_ddir'}), loc=1) \
				.replace(to_replace=-1, value=0).astype(int))
			tdata = pd_split_ternary_to_binary(df_del_midx_level(\
				fd['d']['pba']['dret']['pba_hoc_hdxret_dret'] \
				.rename(columns={-1:'pba_hoc_hdxret_dret'}), loc=1))
		else:
			ld, td = get_xg_label_target_dfs(self.asset_name,
				overwrite_cache=self.overwrite_cache)
			ldata, tdata = get_hardcoded_label_target_dfs(ld, td, self.ldata_name)

		self.data = get_common_interval_data(fdata, ldata, tdata,
			interval=self.interval)

	def setup(self):
		"""
		Split the data into {train, val, test} splits and conert to numpy arrays.
		TODO I can do work done by get_dataloader here (call window_shifted)
		"""
		self.train, self.val, self.test = zip(*map(pd_to_np_tvt, self.data))
		# self.obs_shape = (self.train[0].shape[1], self.t_params['window_size'], \
		# 	self.train[0].shape[-1])	# (Channels, Window, Hours or Window Obs)

		shapes = np.asarray(tuple(map(lambda tvt: tuple(map(np.shape, tvt)), \
			(self.train, self.val, self.test))))
		assert all(np.array_equal(a[:, 1:], b[:, 1:]) for a, b in pairwise(shapes)), \
			'feature, label, target shapes must be identical across splits'
		assert all(len(np.unique(mat.T[0, :]))==1 for mat in shapes), \
			'first dimension (N) must be equal in each split for all (f, l, t) tensors'

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

	def get_benchmarks(self):
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
		bench_strats = (BuyAndHoldStrategy(), OptimalStrategy())
		for pfx, flt in zip(('train', 'val', 'test'), \
			(self.flt_train, self.flt_val, self.flt_test)):
			l = np.sum(flt[1], axis=(1, 2), keepdims=False)
			for stat in bench_stats:
				bench_dict.update(stat(l, pfx))

			t = np.sum(flt[2], axis=(1, 2), keepdims=False)
			t = torch.tensor(t, dtype=torch.float32, requires_grad=False)
			for strat in bench_strats:
				strat.update(None, None, t)
				vals = strat.compute(pfx=pfx)
				bench_dict.update(vals)
				strat.reset()
		return rectify_json(bench_dict)

