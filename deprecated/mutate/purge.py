"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common_util import RECON_DIR
from recon.common import dum


def get_train_times(obs_times, test_times):
	"""
	Given test_times, find the times of the training observations.

	Args:
		obs_times (pd.Series): series of observation start and end times where
			index -> time of observation start, value -> time of observation end
		test_times: times of testing observations

	Lopez De Prado, Advances in Financial Machine Learning (p. 106)
	"""
	train = obs_times.copy(deep=True)

	for start, end in test_times.iteritems():
		partial_overlap_search_dict = {
			'index': ('ine', start, end), # partial overlap (train starts within a test)
			'value': ('ine', start, end)  # partial overlap (train ends within a test)
		}

		full_overlap_search_dict = {
			'index': ('lte', start),	  # full overlap (train evelops test)
			'value': ('gte', end)
		}

		partial_overlap = build_query(partial_overlap_search_dict, join_method='any')
		full_overlap = build_query(full_overlap_search_dict, join_method='all')
		final_query = ' or '.join([partial_overlap, full_overlap])

		train = train.drop(query_df(train, final_query))

	return train


def get_embargo_times(times, pct_embargo):
	"""
	Get embargo times for each bar

	Lopez De Prado, Advances in Financial Machine Learning (p. 108)
	"""

	step = int(times.shape[0]*pct_embargo)

	if (step == 0):
		embargo = pd.Series(times, index=times)
	else:
		embargo = pd.Series(times[step:], index=times[:-step])
		embargo = embargo.append(pd.Series(times[-1], index=times[-step:]))
	return embargo
