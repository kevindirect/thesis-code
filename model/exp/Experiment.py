"""
Kevin Patel
"""

import sys
import os
from itertools import product
import logging

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from common_util import MODEL_DIR
from model.common import MODELS_DIR, EXP_DIR, RESULT_DIR, ERROR_CODE, TRIALS_COUNT, TEST_RATIO, VAL_RATIO
from model.model_util import prepare_transpose_data, prepare_masked_labels
from recon.dataset_util import prep_dataset, gen_group


class Experiment:
	"""
	Abstract base class of all experiment subclasses.
	Experiments contain:
		- A Model
		- A Dataspace
		- Code to perform any necessary data preprocessing prior to running the model
		- Code to run an experiment and optimize the model over its data and hyperparameters
		- Code that dumps the experiment trial history, metadata, and results to disk
	"""
	def __init__(self, model, dataset_name, assets=None):
		"""
		
		Args:
			num_trials (int): Number of hyperopt trials to run
			data_search ('grid' | 'hyper'): Search strategy over data space
			test_ratio (float): ratio of dataset to use for holdout testing
			val_ratio (float): ratio of remaining (non-holdout data) to use for validation
			
		"""
		self.num_trials = TRIALS_COUNT
		self.data_search = 'grid'
		self.assets = assets
		self.dataset_name = dataset_name
		self.model = model()

		self.load_dataset()


	def load_dataset(self):
		dataset_dict = load_json(self.dataset_name, dir_path=DATASET_DIR)
		self.dataset = prep_dataset(dataset_dict, assets=self.assets, filters_map=None)

	def yield_data(self, feat_prep_fn, label_prep_fn):
		"""
		Every Experiment will implement this method to be the top level data yield function.
		"""
		pass

	def yield_1_to_1(self, feat_prep_fn, label_prep_fn):
		"""
		Yield data from the dataset as a product of all feature and label columns.
		"""
		for paths, dfs in gen_group(self.dataset):
			fpaths, lpaths, rpaths = paths
			features, labels, row_masks = dfs

			logging.debug('fpaths: {}'.format(str(fpaths)))
			logging.debug('lpaths: {}'.format(str(lpaths)))
			logging.debug('rpaths: {}'.format(str(rpaths)))

			masked_labels = prepare_masked_labels(labels, ['bool'], labs_filter)

			for feat_col, label_col in product(features.columns, masked_labels.columns):
				feature = feat_prep_fn(features.loc[:, [feat_col]], row_masks).dropna(axis=0, how='all')
				label = delayed(label_prep_fn)(masked_labels.loc[:, label_col]).dropna()
				
				yield feature, label

			# for feat_idx, label_idx in product(*dataset_grid.values()):
			# 	final_feature = prepare_transpose_data(features.iloc[:, [feat_idx]], row_masks).dropna(axis=0, how='all')
			# 	shifted_label = delayed(shift_label)(masked_labels.iloc[:, label_idx]).dropna()
			# 	pos_label, neg_label = delayed(pd_binary_clip, nout=2)(shifted_label)
			# 	f, lpos, lneg = delayed(pd_common_index_rows, nout=3)(final_feature, pos_label, neg_label).compute()

			# 	logging.info('pos dir model experiment')
			# 	run_trials(ThreeLayerBinaryFFN, f, lpos)

			# 	logging.info('neg dir model experiment')
			# 	run_trials(ThreeLayerBinaryFFN, f, lneg)


	def yield_df_to_1(self, feat_prep_fn, label_prep_fn):
		"""
		Yield data from the dataset, map each feature df to each label column.
		"""
		for paths, dfs in gen_group(self.dataset):
			fpaths, lpaths, rpaths = paths
			features, labels, row_masks = dfs

			logging.debug('fpaths: {}'.format(str(fpaths)))
			logging.debug('lpaths: {}'.format(str(lpaths)))
			logging.debug('rpaths: {}'.format(str(rpaths)))

			masked_labels = prepare_masked_labels(labels, ['bool'], labs_filter)
			features = feat_prep_fn(features, row_masks).dropna(axis=0, how='all')

			for label_col in masked_labels.columns:
				label = delayed(label_prep_fn)(masked_labels.loc[:, label_col]).dropna()
				yield features, label


	def run_trials(self, model, f, l):
		trials = Trials()
		obj = self.model.make_const_data_objective(f, l)
		best = fmin(obj, self.model.get_space(), algo=tpe.suggest, max_evals=self.num_trials, trials=trials)

		best_params = self.model.params_idx_to_name(best)
		bad = self.model.get_bad_trials()

		print('best idx: {}'.format(best))
		print('best params: {}'.format(best_params))
		if (bad > 0):
			print('bad trials: {}'.format(bad))

		return best_params


	def features_pipe(self):
		pass

	def labels_pipe(self):
		pass



	def initialize(self):
		pass


	def run(self):
		pass


	def dump(self):
		pass

