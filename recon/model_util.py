# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common_util import remove_dups_list, list_get_dict
from recon.common import dum

class MyPipeline(Pipeline):
	"""
	Enhanced Pipeline Class
	Lopez De Prado, Advances in Financial Machine Learning (p. 131)
	"""
	def fit(self, X, y, sample_weight=None, **fit_params):
		if (sample_weight is not None):
			fit_params[self.steps[-1][0]+'__sample_weight'] = sample_weight
		return super(MyPipeline, self).fit(X, y, **fit_params)

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
		yield feats[train_index], feats[test_index], lab[train_index], lab[test_index]

