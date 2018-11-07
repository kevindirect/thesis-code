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
	def __init__(self):
		"""
		
		Args:
			num_trials (int): Number of hyperopt trials to run
			data_search ('grid' | 'hyper'): Search strategy over data space
			test_ratio (float): ratio of dataset to use for holdout testing
			val_ratio (float): ratio of remaining (non-holdout data) to use for validation
			
		"""
		self.num_trials = TRIALS_COUNT
		self.data_search = 'grid'

		self.
	

	def run():
		pass

	def dump():
		pass
