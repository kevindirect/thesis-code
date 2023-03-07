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
from data.common import PROC_NAME, VENDOR_NAME
from data.window_util import overlap_win_preproc_3d, windowed_ctx_tgt


class XGDataModule(pl.LightningDataModule):
	"""
	Experiment Group Data Module
	Defines the data pipeline from experiment group data on disk to
	pytorch dataloaders which can be used for model training.

	There are three steps to this process:
		1. prepare_data()
		2. setup()
		3. {train, val, test}_dataloader()

	Note: "target" here refers to the regression target, it is not used in the sense
		of neural process context/target observation sets.
"""

	def __init__(self, params_d, proc_name=PROC_NAME, vendor_name=VENDOR_NAME, asset_name="SPX",
		feature_name="price,ivol", target_name="rvol_1day_r_1min_std", return_name="R_1day"):
		super().__init__()
		self.params_d = params_d
		self.proc_name = proc_name
		self.vendor_name = vendor_name
		self.asset_name = asset_name
		self.feature_name = feature_name
		self.target_name = target_name
		self.return_name = return_name
		self.name = f"{asset_name}{sep}{target_name}{sep}{feature_name}"
		self.ddir = f"{DATA_DIR}{self.proc_name}{sep}{self.vendor_name}{sep}{self.asset_name}"
		self.target_names = None
		self.fshape = None
		if (self.vendor_name == "frd"):
			self.day_size = 391 # number of data points per trading day
		if (self.params_d["forecast_delta"]==0):
			logging.warning("Forecast delta is '0', labels will not be shifted forward in time.")

	def prepare_feature(self, split="train"):
		"""
		outputs data shaped like (n, C, H, W).
			n - index
			C - channels (financial instrument/ticker)
			H - columns (open, high, low, close, etc)
			W - window (time sequence in index aggregation period)
		"""
		dfs = [load_df(name, f"{self.ddir}/{split}/feature").set_index("datetime")
			for name in self.feature_name.split(',')]
		assert all((df.index == dfs[0].index).all() for df in dfs)
		arrs = [np_truncate_vstack_2d(df.to_numpy(), self.day_size) for df in dfs]
		assert all(arr.shape == arrs[0].shape for arr in arrs)
		return np.stack(arrs, axis=1)

	def prepare_target(self, split="train"):
		"""
		outputs data shaped like (n,).
			n - index
		"""
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
		Operations here should not depend on params_d,
		so that params_d can be modified without this method
		needing to be caled again.
		"""
		self.data = NestedDefaultDict()
		for split in ["train", "val", "test"]:
			np_feature = self.prepare_feature(split)
			np_index, np_return, np_target = self.prepare_target(split)
			assert np_feature.shape[0] == np_index.shape[0]
			self.data[[split, "index"]] = np_index
			self.data[[split, "feature"]] = np_feature
			self.data[[split, "return"]] = np_return
			self.data[[split, "target"]] = np_target

	def setup(self, stage=None):
		"""
		Apply moving window to the features,
		make the TensorDatasets and samplers used to later create DataLoaders.
		Depends on params_d.
		"""
		self.index, self.dataset, self.sampler = {}, {}, {}
		if (self.params_d["standardize"]):
			self.standardize("target")

		for split in ["train", "val", "test"]:
			windowed = overlap_win_preproc_3d((
					self.data[[split, "index"]],
					self.data[[split, "feature"]],
					self.data[[split, "target"]],
					self.data[[split, "return"]]
				),
				self.params_d["window_size"],
				same_dims=True
			)
			self.index[split] = windowed[0]
			self.dataset[split] = self.get_meta_dataset(windowed, split)

	def get_fshape(self):
		"""
		Shape of each feature observation.
		Depends on params_d.

		If feature tensor is shaped (n, C, H, W) this will be (C, H, W), where:
			* C: channel (data source)
			* H: data column (time series)
			* W: data row (lookback window)
		"""
		if (isnt(self.fshape)):
			_fshape = list(self.data[["train", "feature"]][0].shape)
			_fshape[-1] *= self.params_d["window_size"]
			self.fshape = tuple(_fshape)
		return self.fshape

	def update(self, new):
		if (self.params_d != new):
			self.params_d = new
			self.fshape = None
			self.setup()

	def get_dataloader(self, split):
		return DataLoader(self.dataset[split],
			batch_size=self.params_d['batch_size'],
			shuffle=self.params_d['shuffle'] and split=='train',
			drop_last=True, # TODO
			num_workers=self.params_d['num_workers'],
			pin_memory=self.params_d['pin_memory']
		)

	train_dataloader = lambda self: self.get_dataloader('train')
	val_dataloader = lambda self: self.get_dataloader('val')
	test_dataloader = lambda self: self.get_dataloader('test')

	def get_target_names(self, split="train"):
		if (isnt(self.target_names)):
			price = load_df("price", f"{self.ddir}/{split}/target").set_index("datetime")
			self.target_names = list(price.columns)
		return self.target_names

	def standardize(self, subset="target", sample_split="train"):
		"""
		Standardize by sample mean, std statistics.
		"""
		sample_mean = np.mean(self.data[[sample_split, subset]])
		sample_std = np.std(self.data[[sample_split, subset]])

		for split in ["train", "val", "test"]:
			self.data[[split, subset]] = (self.data[[split, subset]] - sample_mean) / sample_std

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
	def get_tensors(data, delta=1):
		"""
		Return tuple of index, features, targets, returns tensor.
		Shift index, feature, target, etc appropriately by delta.

		Note that the index of t+delta is stored (ie the index of the label),
		This means that predictions are indexed by the timestamp they predict for
		and not the one they were derived from.
		If the predictions are used as a feauture to a stacked model, the data
		should not be shifted first. The indices just need to be joined with
		the stacked model label.

		Args:
			data (tuple): tuple of numpy arrays, features are the first element

		Returns:
			tuple of tensors
		"""
		i = torch.arange(len(data[0][delta:]), requires_grad=False) # int index to avoid storing datetime in tensor
		f = torch.tensor(data[1][:data[1].shape[0]-delta], dtype=torch.float32, requires_grad=False)
		t = torch.tensor(data[2][delta:], dtype=torch.float32, requires_grad=False)
		r = torch.tensor(data[3][delta:], dtype=torch.float32, requires_grad=False)
		assert all(d.shape[0]==i.shape[0] for d in [f, t, r])
		return i, f, t, r

	def get_dataset(self, data):
		"""
		Dataset with tensors shaped (n, *), where:
			* n: dataset size (number of observations)
			* *: observation dimensions
		"""
		return TensorDataset(*self.get_tensors(data, delta=self.params_d["forecast_delta"]))

	def get_meta_dataset(self, data, split):
		"""
		Meta Dataset with tensors shaped (n, e, *), where:
			* n: meta dataset size (number of episodes)
			* e: episode size (number of observations)
			* *: observation dimensions
		"""
		i, f, t, r = self.get_tensors(data, delta=self.params_d["forecast_delta"])
		train_mode = split=='train'
		step_size = self.params_d['step_size'] if (train_mode) else self.params_d['context_size']
		ctx, tgt = windowed_ctx_tgt(
			i,
			self.params_d['context_size'],
			self.params_d['target_size'],
			step_size or self.params_d['context_size'],
			self.params_d['overlap_size'],
			self.params_d['resample_context'] and train_mode
		)
		assert len(ctx)==len(tgt)
		return TensorDataset(i[ctx], f[ctx], t[ctx], r[ctx], i[tgt], f[tgt], t[tgt], r[tgt])

