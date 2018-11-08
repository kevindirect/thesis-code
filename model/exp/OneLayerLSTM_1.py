"""
Kevin Patel
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from common_util import MODEL_DIR
from model.common import MODELS_DIR, EXP_DIR, RESULT_DIR, ERROR_CODE, TRIALS_COUNT, TEST_RATIO, VAL_RATIO


class OneLayerLSTM_1(Experiment):
	"""
	Abstract base class of all experiment subclasses.
	Experiments contain:
		- A Model
		- A Dataspace
		- Code to perform any necessary data preprocessing prior to running the model
		- Code to run an experiment and optimize the model over its data and hyperparameters
		- Code that dumps the experiment trial history, metadata, and results to disk
	"""
	def __init__(self, dataset_name, assets=None):
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
		self.model = None

		self.load_dataset()
		super(OneLayerLSTM_1, self).__init__(dataset_name, assets=assets)

	def yield_data(self, feat_prep_fn=prepare_transpose_data, label_prep_fn=shift_label):

		for feature, label in self.yield_1_to_1(prepare_transpose_data, shift_label):
			pos_label, neg_label = delayed(pd_binary_clip, nout=2)(shifted_label)
			f, lpos, lneg = delayed(pd_common_index_rows, nout=3)(final_feature, pos_label, neg_label).compute()

			yield f, lpos, lneg


	def run():
		pass

	def dump():
		pass
