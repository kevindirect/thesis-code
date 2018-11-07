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
from recon.dataset_util import prep_dataset, prep_labels, gen_group


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
	def __init__(self, assets=None):
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


	def load_dataset(self, dataset_name):
		dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
		self.dataset = prep_dataset(dataset_dict, assets=self.assets, filters_map=None)


	def yield_data(self):
		for paths, dfs in gen_group(self.dataset):
			fpaths, lpaths, rpaths = paths
			features, labels, row_masks = dfs
			# asset = fpaths[0]
			logging.info('fpaths: {}'.format(str(fpaths)))
			logging.info('lpaths: {}'.format(str(lpaths)))
			logging.info('rpaths: {}'.format(str(rpaths)))

			masked_labels = prepare_masked_labels(labels, ['bool'], labs_filter)

			for feat_idx, label_idx in product(*dataset_grid.values()):
				final_feature = prepare_transpose_data(features.iloc[:, [feat_idx]], row_masks).dropna(axis=0, how='all')
				shifted_label = delayed(shift_label)(masked_labels.iloc[:, label_idx]).dropna()
				pos_label, neg_label = delayed(pd_binary_clip, nout=2)(shifted_label)
				f, lpos, lneg = delayed(pd_common_index_rows, nout=3)(final_feature, pos_label, neg_label).compute()

				logging.info('pos dir model experiment')
				run_trials(ThreeLayerBinaryFFN, f, lpos)

				logging.info('neg dir model experiment')
				run_trials(ThreeLayerBinaryFFN, f, lneg)


def run_trials(model_exp, features, label):
	exp = model_exp()
	trials = Trials()
	obj = exp.make_const_data_objective(features, label)
	best = fmin(obj, exp.get_space(), algo=tpe.suggest, max_evals=50, trials=trials)
	best_params = exp.params_idx_to_name(best)
	bad = exp.get_bad_trials()

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

