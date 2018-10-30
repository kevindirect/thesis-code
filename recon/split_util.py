# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit

from common_util import RECON_DIR
from recon.common import dum


""" ********** VALUE REMAP ********** """
def pd_binary_clip(pd_obj, thresh=0, return_abs=True):
	"""
	Given a pd_obj split it into two by clipping above and below threshold and returning
	the two result pd_objects.
	"""
	keep_above = pd_obj.clip(lower=thresh, upper=None)
	keep_below = pd_obj.clip(lower=None, upper=thresh)

	if (return_abs):
		if (thresh < 0):
			keep_above = keep_above.abs()
		keep_below = keep_below.abs()

	return keep_above, keep_below


""" ********** TRAIN/TEST SPLITS ********** """
def get_train_test_split(feats, lab, test_ratio=.8, to_np=True):
	"""
	Return a basic train/test split.

	Args:
		feats (pd.DataFrame or ndarray): features dataframe
		lab (pd.Series or ndarray): label series
		test_ratio (float ∈ [0, 1]): proportion of total data used for test 
		to_np (bool): boolean determines whether to convert input to numpy types

	Returns:
		Tuple of (feats_train, feats_test, lab_train, lab_test)
	"""
	if (to_np):
		return train_test_split(feats.values, lab.values, test_size=test_ratio, shuffle=False)
	else:
		return train_test_split(feats, lab, test_size=test_ratio, shuffle=False)

def gen_time_series_split(feats, lab, num_splits=5, max_train=None):
	tscv = TimeSeriesSplit(n_splits=num_splits, max_train_size=max_train)

	for train_index, test_index in tscv.split(feats):
		yield feats[train_index], feats[test_index], lab[train_index], lab[test_index]


""" ********** CROSS VALIDATION SPLITS ********** """
DEFAULT_CV_TRANSLATOR = {
	"KFold": KFold,
	"TimeSeriesSplit": TimeSeriesSplit,
	"__name": "DEFAULT_CV_TRANSLATOR"
}

def translate_cv(cv_name, cv_params=None, translator=DEFAULT_CV_TRANSLATOR):
	cv_constructor = translator.get(cv_name, None)

	if (cv_constructor is None):
		raise ValueError(cv_name, 'does not exist in', translator['__name'])
	else:
		if (cv_params is not None):
			return cv_constructor(**cv_params)
		else:
			return cv_constructor()

def extract_cv_splitter(dictionary):
	"""
	Converts a passed pipeline dictionary into a sklearn Pipeline object and parameter grid
	"""
	cv_splitter = translate_cv(dictionary['name'], cv_params=dictionary['params'])
	
	return cv_splitter