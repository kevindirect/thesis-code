# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common_util import remove_dups_list, list_get_dict
from recon.common import dum


def get_train_test_split(feats, lab, train_ratio=.8, to_np=True):
	"""
	Returns:
		train test split - feats_train, feats_test, lab_train, lab_test
	"""
	if (to_np):
		return train_test_split(feats.values, lab.values, train_size=train_ratio, shuffle=False)
	else:
		return train_test_split(feats, lab, train_size=train_ratio, shuffle=False)


def gen_time_series_split(feats, lab, num_splits=5, max_train=None):
	tscv = TimeSeriesSplit(n_splits=num_splits, max_train_size=max_train)

	for train_index, test_index in tscv.split(feats):
		yield feats[train_index], lab[train_index], feats[test_index], lab[test_index]


