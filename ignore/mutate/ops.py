# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize
from dask import delayed

from common_util import search_df, get_subset, pd_to_np
from mutate.common import dum
# reconstruct...


""" STATIONARITY OPERATIONS """


""" ROLLING OPERATIONS """

RollType = Enum('RollType', 'MEAN STD')

def simple_moving(ser, roll_type=RollType.MEAN, window=8, min_periods=None, win_type=None):
	"""
	Return simple moving average or standard deviation of ser.
	"""
	if (roll_type == RollType.MEAN):
		return ser.rolling(window, min_periods=min_periods, win_type=win_type).mean()
	elif (roll_type == RollType.STD):
		return ser.rolling(window, min_periods=min_periods, win_type=win_type).std()


def exponential_moving(ser, roll_type=RollType.MEAN, com=None, span=8, halflife=None, alpha=None, min_periods=None):
	"""
	Return exponential moving average or standard deviation of ser.

	Documentation from http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows site:
	
		One must specify precisely one of span, center of mass, half-life and alpha to the EW functions
		Span corresponds to what is commonly called an “N-day EW moving average”.
		Center of mass has a more physical interpretation and can be thought of in terms of span: c=(s−1)/2.
		Half-life is the period of time for the exponential weight to reduce to one half.
		Alpha specifies the smoothing factor directly.
	"""
	if (roll_type == RollType.MEAN):
		return ser.ewm(com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods).mean()
	elif (roll_type == RollType.STD):
		return ser.ewm(com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods).std()



# def vol_estimates(df):
# 	"""
# 	Return a df of volatility estimates
# 	"""
# 	REQUIRED_COLS = ['pba_open', 'pba_high', 'pba_low', 'pba_close',
# 					'vol_open', 'vol_high', 'vol_low', 'vol_close']

# 	assert(set(REQUIRED_COLS) <= set(df.columns)) # Assert df columns is subset of required

# 	vol_est_df = pd.DataFrame(df.index)

# 	diff = pd.Series(subtract(df['pba_close'], df['pba_open']))
# 	pct_diff = diff / df['pba_close']

# 	# Prev price diff
# 	vol_est_df['prev_diff'] = diff.shift(1)

# 	# Prev price pct diff
# 	vol_est_df['prev_pct_diff'] = pct_diff.shift(1)

# 	# Prev price spread
# 	vol_est_df['prev_spread'] = (df['pba_high'] - df['pba_low']).shift(1)

# 	# Prev IV
# 	vol_est_df['prev_vol_diff'] = (df['vol_close'] - df['vol_open']).shift(1)

# 	# Prev IV pct_change
# 	vol_est_df['prev_vol_pct_diff'] = vol_est_df['prev_vol_diff'] / df['vol_close'].shift(1)

# 	# Prev IV spread
# 	vol_est_df['prev_vol_spread'] = (df['vol_high'] - df['vol_low']).shift(1)

# 	return vol_est_df



