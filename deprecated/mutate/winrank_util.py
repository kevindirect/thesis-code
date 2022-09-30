"""
Kevin Patel
"""
import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import benchmark
from mutate.common import dum


# Window function examples
fixed_window = lambda ser, thresh=20: len(ser) >= thresh
variance_thresh = lambda ser, thresh=.01: ser.var(skipna=True) >= thresh

# Rank function examples
min_max_rank = lambda ser: 2 * ((ser.iloc[-1]-ser.min()) / (ser.max()-ser.min())) - 1
ordinal_rank = lambda ser, normalize=True: ser.rank(numeric_only=True, ascending=True, pct=normalize).iloc[-1]
percentile_rank = lambda ser: (ser.iloc[-1]-ser.min()) / (ser.max()-ser.min())


def reverse_window_rank(pd_ser, win_fn=partial(variance_thresh, thresh=.1), rank_fn=percentile_rank):
	"""
	Return pandas series or dataframe where each value is mapped to it's rank within a window

	Args:
		pd_ser (pd.Series): series to work on
		win_fn (function): function that determines the window size, of the form fn(test_series) -> bool
		rank_fn (function): how to rank the point within the window, of the form fn(window_series) -> numeric

	Returns:
		pd_obj with values ranked according to passed parameters by a reverse moving window
	"""
	pd_ser = pd_ser.dropna(inplace=False)
	rank_ser = pd.Series(index=pd_ser.index)

	for cur_point in range(pd_ser.shape[0]-1, -1, -1):
		cur_rank = None
		for win_start in range(cur_point-1, -1, -1):
			win = pd_ser.iloc[win_start:cur_point]
			if (win_fn(win)):
				cur_rank = rank_fn(win)
				break
		rank_ser[cur_point] = cur_rank

	return rank_ser






