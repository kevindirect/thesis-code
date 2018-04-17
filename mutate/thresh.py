# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from numba import jit, vectorize
# from dask import delayed, compute

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.ops import *


# ********** THRESH TYPES **********
_spread_thresh = lambda f, s: abs(f - s)			# spread: abs arithmetic spread -> |fast - slow|
_ansr_thresh = lambda f, s: abs((f / s) - 1)		# ansr: abs net simple return 	-> |(fast / slow) - 1|
_alog_thresh = lambda f, s: abs(np.log(f / s))		# alog: abs log gross return 	-> |ln(fast / slow)|


# ********** THRESH SHIFTED ON AGG_FREQ **********
def get_thresh_af(intraday_df, thresh_type='ansr', original_freq=DT_HOURLY_FREQ, agg_freq=DT_BIZ_DAILY_FREQ, pfx=''):
	"""
	Return thresh estimates based on pervious aggregation period, shifted on aggregation frequency.
	
	Args:
		intraday_df (pd.DataFrame): intraday price dataframe with two columns (slow, fast)
		thresh_type (String): threshold type
		pfx (String, optional): prefix to all column names
		original_freq: freq of the original data
		agg_freq: freq to use for groupby aggregation and shift
	
	Returns:
		Return pd.DataFrame with derived columns
	"""
	_cname = lambda s: '_'.join([pfx, s, thresh_type])

	if (thresh_type == 'spread'):
		thresh_fun = _spread_thresh
	elif (thresh_type == 'ansr'):
		thresh_fun = _ansr_thresh
	elif (thresh_type == 'alog'):
		thresh_fun = _alog_thresh

	derived = pd.DataFrame(index=intraday_df.index)
	derived['slow'] = intraday_df.iloc[:, 0]
	derived['fast'] = intraday_df.iloc[:, 1]
	derived['thresh'] = thresh_fun(derived['fast'], derived['slow'])
	gb = derived.groupby(pd.Grouper(freq=agg_freq))
	# roll = derived.rolling(agg_freq)

	# FIXED WINDOW (CURRENT PERIOD ONLY)
	# average
	derived[_cname('avg')] = gb['thresh'].transform(pd.Series.mean).shift(freq=agg_freq)

	# standard deviation
	derived[_cname('std')] = gb['thresh'].transform(pd.Series.std).shift(freq=agg_freq)

	# median
	derived[_cname('med')] = gb['thresh'].transform(pd.Series.median).shift(freq=agg_freq)
	
	# largest
	derived[_cname('max')] = gb['thresh'].transform(pd.Series.max).shift(freq=agg_freq)

	# smallest
	derived[_cname('min')] = gb['thresh'].transform(pd.Series.min).shift(freq=agg_freq)

	# second-to-last of previous day
	derived[_cname('sec')] = gb['thresh'].transform(lambda x: x.iat[len(x)-2]).shift(freq=agg_freq)
	
	# final of previous day
	derived[_cname('fin')] = gb['thresh'].transform(pd.Series.last, original_freq).shift(freq=agg_freq)

	# whole of previous day
	last_fast = gb['fast'].transform(pd.Series.last, original_freq)
	first_slow = gb['slow'].transform(pd.Series.first, original_freq)
	whole = thresh_fun(last_fast, first_slow)
	derived[_cname('whl')] = whole.shift(freq=agg_freq)


	# # ROLLING (PAST INCLUDED) at agg_freq aggregation period
	# # agg_freq moving average
	# derived[_cname('afma')] = roll.mean()

	# # agg_freq exponential moving average
	# derived[_cname('afema')]

	# # agg_freq moving standard deviation
	# derived[_cname('afmsd')] = roll.std()

	# # agg_freq exponential moving standard deviation
	# derived[_cname('afesd')]

	# # agg_freq moving median
	# derived[_cname('afmed')] = roll.median()

	# # agg_freq moving max
	# derived[_cname('afmax')] = roll.max()

	# # agg_freq moving min
	# derived[_cname('afmin')] = roll.min()


	# # ROLLING (PAST INCLUDED) at original_freq aggregation period
	# # original_freq moving average
	# derived[_cname('ofma')]

	# # original_freq exponential moving average
	# derived[_cname('ofema')]

	# # original_freq moving standard deviation
	# derived[_cname('ofmsd')]

	# # original_freq exponential moving standard deviation
	# derived[_cname('ofesd')]

	# # original_freq moving median
	# derived[_cname('ofmed')]

	# # original_freq moving max
	# derived[_cname('ofmax')]

	# # original_freq moving min
	# derived[_cname('ofmin')]

	
	# Deal with days where previous day has less hours than current
	derived = derived.groupby(pd.Grouper(freq=agg_freq)).fillna(method='ffill')

	return derived


# ********** THRESH SHIFTED ON ORIGINAL_FREQ **********
def get_thresh_of(intraday_df, thresh_type='ansr', original_freq=DT_HOURLY_FREQ, agg_freq=DT_BIZ_DAILY_FREQ, pfx=''):
	"""
	Return thresh estimates based on current aggregation period, shifted on original frequency.
	
	Args:
		intraday_df (pd.DataFrame): intraday price dataframe with two columns (slow, fast)
		thresh_type (String): threshold type
		pfx (String, optional): prefix to all column names
		original_freq: freq of the original data
		agg_freq: freq to use for groupby and shift
	
	Returns:
		Return pd.DataFrame with derived columns
	"""
	_cname = lambda s: '_'.join([pfx, s, thresh_type])

	if (thresh_type == 'spread'):
		thresh_fun = _spread_thresh
	elif (thesh_type == 'ansr'):
		thresh_fun = _ansr_thresh
	elif (thresh_type == 'log'):
		thresh_fun = _log_thresh

	derived = pd.DataFrame(index=intraday_df.index)
	derived['slow'] = intraday_df.iloc[:, 0]
	derived['fast'] = intraday_df.iloc[:, 1]
	derived['thresh'] = thresh_fun(derived['fast'], derived['slow'])
	gb = derived.loc[:, 'thresh'].groupby(pd.Grouper(freq=agg_freq))

	# FIXED WINDOW (CURRENT PERIOD ONLY)
	# latest
	derived[_cname('lst')] = gb['thresh'].transform(pd.Series.last, original_freq).shift(freq=original_freq)

	# expanding average
	derived[_cname('xma')] = gb['thresh'].transform(lambda ser: ser.expanding.mean())

	# expanding standard deviation
	derived[_cname('xsd')] = gb['thresh'].transform(lambda ser: ser.expanding.std())

	# expanding max
	derived[_cname('xmax')] = gb['thresh'].transform(lambda ser: ser.expanding.max())
	
	# expanding min
	derived[_cname('xmin')] = gb['thresh'].transform(lambda ser: ser.expanding.min())


	# # exponential average
	# derived[_cname('ema')]

	# # exponential standard deviation
	# derived[_cname('exesd')] = gb['thresh'].transform(pd.Series.std).shift(freq=agg_freq)


	# # ROLLING (PAST INCLUDED) at original_freq aggregation period
	# # original_freq moving average
	# derived[_cname('ofma')]

	# # original_freq exponential moving average
	# derived[_cname('ofema')]

	# # original_freq moving standard deviation
	# derived[_cname('ofmsd')]

	# # original_freq exponential moving standard deviation
	# derived[_cname('ofesd')]

	# # original_freq moving median
	# derived[_cname('ofmed')]

	# # original_freq moving max
	# derived[_cname('ofmax')]

	# # original_freq moving min
	# derived[_cname('ofmin')]


	
	# Deal with days where previous day has less hours than current
	# derived = derived.groupby(pd.Grouper(freq=agg_freq)).fillna(method='ffill')

	return derived

