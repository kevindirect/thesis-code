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

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum

# TODO: fracdiff, vth, single series thresholds

"""
THRESHOLD TYPE (Transform on price series to make it a stationary return series)
	* spread: arithmetic spread (period 1 differencing)
	* return: net simple return
	* logret: log gross return
	* frac: fractional differentiation

TIME HORIZON (Allowable data window to make threshold from)
	* FTH: Fixed Time Horizon - Uses the latest aggregation period
	* VTH: Variable Time Horizon - Uses the latest aggregation period and previous periods

THRESH TRANSFORMS (Types of transforms on return series to make a threshold)
	* static: Uses all data starting from the time horizon to current
	* moving: Uses all data starting from k periods to current
		- rolling: simple aggregation
		- ewm: exponentially weighted aggregation
	* expanding: Uses all data to current

"""


# ********** THRESHOLD TYPES **********
_spread_thresh = lambda f, s: f - s				# spread: arithmetic spread -> fast - slow
_return_thresh = lambda f, s: f / s - 1			# return: net simple return -> fast / slow - 1
_logret_thresh = lambda f, s: np.log(f / s)		# logret: log gross return 	-> ln(fast / slow)


# ********** FIXED TIME HORIZON **********
def get_thresh_fth(intraday_df, thresh_type='return', shift=False, src_data_pfx='',
	org_freq=DT_HOURLY_FREQ, agg_freq=DT_BIZ_DAILY_FREQ, shift_freq=DT_BIZ_DAILY_FREQ):
	"""
	Return thresh estimates.

	Args:
		intraday_df (pd.DataFrame): intraday price dataframe with two columns (fast, slow)
		thresh_type (String): threshold type
		shift (boolean): whether or not to shift by 'shift_freq'
		src_data_pfx (String, optional): data source indentifiying string
		org_freq: freq of the original data
		agg_freq: freq to use for groupby aggregation
		shift_freq: freq to use for shift ∈ {org_freq, agg_freq}
	
	Returns:
		Return pd.DataFrame with derived columns
	"""
	th =  'fth'
	sf = 'af' if (shift_freq == agg_freq) else 'of'
	_cname = lambda sfx: '_'.join([th, sf, src_data_pfx, thresh_type, sfx])

	if (thresh_type == 'spread'):
		thresh_fun = _spread_thresh
	elif (thresh_type == 'return'):
		thresh_fun = _return_thresh
	elif (thresh_type == 'logret'):
		thresh_fun = _logret_thresh

	derived = pd.DataFrame(index=intraday_df.index)
	derived['fast'] = intraday_df.iloc[:, 0]
	derived['slow'] = intraday_df.iloc[:, 1]
	derived['thresh'] = thresh_fun(derived['fast'], derived['slow'])
	derived['abs_thresh'] = abs(derived['thresh'])
	# orig_cols = list(derived.columns)
	gb = derived.groupby(pd.Grouper(freq=agg_freq))

	if (shift_freq == agg_freq):
		# static average
		derived[_cname('avg')] = gb['thresh'].transform(pd.Series.mean)

		# static standard deviation
		derived[_cname('std')] = gb['thresh'].transform(pd.Series.std)

		# exponentially weighted average
		derived[_cname('eavg')] = gb['thresh'].transform(lambda s: s.ewm(span=len(s)).mean())

		# exponentially weighted standard deviation
		derived[_cname('estd')] = gb['thresh'].transform(lambda s: s.ewm(span=len(s)).std())

		# static median
		derived[_cname('med')] = gb['thresh'].transform(pd.Series.median)
		
		# static largest
		derived[_cname('max')] = gb['thresh'].transform(pd.Series.max)

		# static smallest
		derived[_cname('min')] = gb['thresh'].transform(pd.Series.min)

		# second-to-last of previous day
		derived[_cname('seclast')] = gb['thresh'].transform(lambda x: x.iat[len(x)-2])
		
		# final of previous day
		derived[_cname('last')] = gb['thresh'].transform(pd.Series.last, org_freq)

		# whole of previous day (can use this to create vth thresholds)
		last_fast = gb['fast'].transform(pd.Series.last, org_freq)
		first_slow = gb['slow'].transform(pd.Series.first, org_freq)
		derived[_cname('whole')] = thresh_fun(last_fast, first_slow)

		# abs static average
		derived[_cname('abs_avg')] = gb['abs_thresh'].transform(pd.Series.mean)

		# abs static standard deviation
		derived[_cname('abs_std')] = gb['abs_thresh'].transform(pd.Series.std)

		# abs exponentially weighted average
		derived[_cname('abs_eavg')] = gb['abs_thresh'].transform(lambda s: s.ewm(span=len(s)).mean())

		# abs exponentially weighted standard deviation
		derived[_cname('abs_estd')] = gb['abs_thresh'].transform(lambda s: s.ewm(span=len(s)).std())

		# abs static median
		derived[_cname('abs_med')] = gb['abs_thresh'].transform(pd.Series.median)
		
		# abs static largest
		derived[_cname('abs_max')] = gb['abs_thresh'].transform(pd.Series.max)

		# abs static smallest
		derived[_cname('abs_min')] = gb['abs_thresh'].transform(pd.Series.min)

		# abs second-to-last of previous day
		derived[_cname('abs_seclast')] = gb['abs_thresh'].transform(lambda x: x.iat[len(x)-2])
		
		# abs final of previous day
		derived[_cname('abs_last')] = gb['abs_thresh'].transform(pd.Series.last, org_freq)

	elif (shift_freq == org_freq):

		# value of previous period
		derived[_cname('actual')] = derived['thresh'] # final version will be shifted

		# expanding average
		derived[_cname('xavg')] = gb['thresh'].transform(lambda ser: ser.expanding().mean())

		# expanding median
		derived[_cname('xmed')] = gb['thresh'].transform(lambda ser: ser.expanding().median())

		# expanding standard deviation
		derived[_cname('xstd')] = gb['thresh'].transform(lambda ser: ser.expanding().std())

		# expanding max
		derived[_cname('xmax')] = gb['thresh'].transform(lambda ser: ser.expanding().max())
		
		# expanding min
		derived[_cname('xmin')] = gb['thresh'].transform(lambda ser: ser.expanding().min())

		# abs expanding average
		derived[_cname('abs_xavg')] = gb['abs_thresh'].transform(lambda ser: ser.expanding().mean())

		# abs expanding median
		derived[_cname('abs_xmed')] = gb['abs_thresh'].transform(lambda ser: ser.expanding().median())

		# abs expanding standard deviation
		derived[_cname('abs_xstd')] = gb['abs_thresh'].transform(lambda ser: ser.expanding().std())

		# abs expanding max
		derived[_cname('abs_xmax')] = gb['abs_thresh'].transform(lambda ser: ser.expanding().max())
		
		# abs expanding min
		derived[_cname('abs_xmin')] = gb['abs_thresh'].transform(lambda ser: ser.expanding().min())

	# if (shift):
	# 	derived_cols = [col_name for col_name in derived.columns if (col_name not in orig_cols)]
	# 	for col_name in derived_cols:
	# 		derived[col_name] = derived[col_name].shift(freq=shift_freq, axis=0)

	# 	# For days where previous day has less hours than current
	# 	derived = derived.groupby(pd.Grouper(freq=agg_freq)).fillna(method='ffill')

	return derived




# ********** VARIABLE TIME HORIZON **********
# TODO: a more sensible way to do this is to base it off the fth version:
#			* shift_freq==agg_freq - > rolling/expanding/ewm on the same fth version 'swhl' column
# 			* shift_freq==orig_freq -> rolling/expanding/ewm on the same fth version 'prev' column
# 				(or take a weighted average of the vth(shift_freq==agg_freq) and intraday portions)

# def get_thresh_vth(intraday_df, thresh_type='return', org_freq=DT_HOURLY_FREQ, agg_freq=DT_BIZ_DAILY_FREQ, shift_freq=DT_BIZ_DAILY_FREQ, per=None, src_data_pfx=''):
# 	"""
# 	Return thresh estimates.
# 	Variable time horizon allows thresholds to go beyond the specified aggregation frequency.

# 	Args:
# 		intraday_df (pd.DataFrame): intraday price dataframe with two columns (slow, fast)
# 		thresh_type (String): threshold type
# 		org_freq: freq of the original data
# 		agg_freq: freq to use for groupby aggregation
# 		shift_freq: freq to use for shift ∈ {org_freq, agg_freq}
# 		src_data_pfx (String, optional): prefix to all column names
	
# 	Returns:
# 		Return pd.DataFrame with derived columns
# 	"""
# 	time_hor = str(per)+str(shift_freq)
# 	th =  'vth(' +time_hor +')'
# 	_cname = lambda s: '_'.join([src_data_pfx, th, s, thresh_type])

# 	if (thresh_type == 'spread'):
# 		thresh_fun = _spread_thresh
# 	elif (thresh_type == 'return'):
# 		thresh_fun = _return_thresh
# 	elif (thresh_type == 'logret'):
# 		thresh_fun = _logret_thresh

# 	derived = pd.DataFrame(index=intraday_df.index)
# 	derived['slow'] = intraday_df.iloc[:, 0]
# 	derived['fast'] = intraday_df.iloc[:, 1]
# 	derived['thresh'] = thresh_fun(derived['fast'], derived['slow'])
# 	gb = derived.groupby(pd.Grouper(freq=agg_freq))
# 	roll = derived.rolling(time_hor) # Rolling object of size 'per' 'agg_freq's
# 	expand = derived.expanding()

# 	# ROLLING
# 	# simple moving average
# 	derived[_cname('s_ma')] = roll.mean().resample(DT_HOURLY_FREQ)

# 	# simple moving standard deviation
# 	derived[_cname('s_std')] = roll.std().resample(DT_HOURLY_FREQ)

# 	# simple moving median
# 	derived[_cname('s_med')] = roll.median().resample(DT_HOURLY_FREQ)
	
# 	# simple moving largest
# 	derived[_cname('s_max')] = roll.max().resample(DT_HOURLY_FREQ)

# 	# simple moving smallest
# 	derived[_cname('s_min')] = roll.min().resample(DT_HOURLY_FREQ)

# 	# EXPANDING
# 	# expanding average
# 	derived[_cname('x_ma')] = expand.mean().resample(DT_HOURLY_FREQ)

# 	# expanding standard deviation
# 	derived[_cname('x_std')] = expand.std().resample(DT_HOURLY_FREQ)

# 	# expanding median
# 	derived[_cname('x_med')] = expand.median().resample(DT_HOURLY_FREQ)
	
# 	# expanding largest
# 	derived[_cname('x_max')] = expand.max().resample(DT_HOURLY_FREQ)

# 	# expanding smallest
# 	derived[_cname('x_min')] = expand.min().resample(DT_HOURLY_FREQ)
	
# 	if (shift_freq == agg_freq):
# 		# # whole of previous day
# 		# last_fast = gb['fast'].transform(pd.Series.last, org_freq)
# 		# first_slow = gb['slow'].transform(pd.Series.first, org_freq)
# 		# whole = thresh_fun(last_fast, first_slow)
# 		# derived[_cname('whl')] = whole.shift(freq=shift_freq)

# 		# # For days where previous day has less hours than current
# 		# derived = derived.groupby(pd.Grouper(freq=agg_freq)).fillna(method='ffill')

# 	elif (shift_freq == org_freq):
# 		ew = derived.ewm(span=per)

# 		# EXPONENTIAL
# 		# simple moving average
# 		derived[_cname('e_ma')] = ew.mean()

# 		# simple moving standard deviation
# 		derived[_cname('e_std')] = ew.std()

# 		# median
# 		derived[_cname('e_med')] = ew.med()
		
# 		# largest
# 		derived[_cname('e_max')] = ew.max()

# 		# smallest
# 		derived[_cname('e_min')] = ew.min()


# 	return derived
