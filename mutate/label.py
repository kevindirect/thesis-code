# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from numba import jit, vectorize, float64
# from dask import delayed, compute

from common_util import DT_BIZ_DAILY_FREQ, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum


UP = 1
DOWN = -1
SIDEWAYS = 0
MarketDir = Enum('MarketDir', 'UP DOWN SIDEWAYS')

# ********** FIXED TIME HORIZON **********
def fth_thresh_label(ser, thresh=0.0, scalar=1.0):
	"""
	Return fixed time horizon label series thresholded by direction.

	Args:
		ser (pd.Series): series of changes to threshold
		thresh (float or pd.Series, ℝ≥0): float or float series to threshold on
		scalar (float, ℝ≥0): threshold multiplier

	Returns:
		pd.Series ∈ {-1, 0, 1}
	"""
	new = ser.copy()
	scaled = thresh * scalar

	new[new >= scaled] = UP
	new[new <= -scaled] = DOWN
	new[~new.isin([UP, DOWN])] = SIDEWAYS

	return new

# ********** INTRADAY TRIPLE BARRIER **********
@vectorize([float64(float64, float64, float64)], nopython=True)
def thresh_break(start, end, thresh):
	change = (end / start) - 1

	return change if (change > thresh or change < -thresh) else 0

def find_touch(group_df, per_shift=1):
	"""
	Return touch found

	Args:
		group_df (pd.DataFrame): dataframe of aggregation period
		per_shift (integer, ℤ): number to shift break period by
	"""
	group_df = group_df.dropna()
	if (group_df.empty):
		return np.NaN

	start_arr = np.array(group_df.loc[:, 'start'].first(DT_HOURLY_FREQ))
	end_arr = np.array(group_df['end'].values)
	thresh_arr = np.array(group_df['thresh'].values)

	stats = {
		"dir": 0,
		"mag": 0,
		"brk": 0,
		"day": end_arr.size
	}

	breaks = thresh_break(start_arr, end_arr, thresh_arr)
	break_ids = np.flatnonzero(breaks)
	
	if (break_ids.size != 0):
		# Change set to first threshold break
		change = breaks[break_ids[0]]
		stats['brk'] = break_ids[0] + per_shift
	else:
		# End of day change, no threshold
		change = (end_arr[-1] / start_arr[0]) - 1

	stats['dir'] = np.sign(change)
	stats['mag'] = abs(change)

	return stats['dir'], stats['mag'], stats['brk'], stats['day']

def intraday_triple_barrier(intraday_df, thresh, scalar={'up': .55, 'down': .45}, agg_freq=DT_BIZ_DAILY_FREQ):
	"""
	Return intraday triple barrier label series.
	
	Args:
		intraday_df (pd.DataFrame): intraday price dataframe
		thresh (String): name of threshold column
		scalar (dict(str: float), ℝ≥0): bull/bear thresh multipliers
	
	Returns:
		Return pd.DataFrame with four columns:
			- 'dir': price direction
			- 'spd': change speed
			- 'brk': period of break (zero if none)
			- 'day': number of trading periods
	"""
	# DF Preprocessing
	col_renames = {
		intraday_df.columns[0]: "start",
		thresh: "thresh"
	}
	num_cols = len(intraday_df.columns)
	if (num_cols == 2):
		intraday_df['end'] = intraday_df[intraday_df.columns[0]]
	elif (num_cols > 2):
		col_renames[intraday_df.columns[1]] = 'end'
	intraday_df.rename(columns=col_renames, inplace=True)
	
	# Threshold scale
	intraday_df['thresh'] = scalar * intraday_df['thresh']

	# Apply
	labels = intraday_df.groupby(pd.Grouper(freq=agg_freq)).apply(find_touch)
	labels = labels.apply(pd.Series)
	labels.columns=['dir','mag', 'brk', 'day']
	return labels



# ********** LOPEZ DE PRADO TRIPLE BARRIER **********

def applyPtS1OnT1(close, events, ptS1, molecule):
	"""
	Adapted from 'applyPtS1OnT1':
		Lopez De Prado, Advances in Financial Machine Learning (p. 45-46)

	Args:
		close (pd.DataFrame): 
		events (String):
		ptS1 ():
		molecule:

	Returns:
		None
	"""

	# apply stop loss/profit taking, if it takes place before t1 (end of event)
	events_ = events.loc[molecule]
	out = events_[['t1']].copy(deep=True)

	if (ptS1[0] > 0):
		pt = ptS1[0] * events_['trgt']
	else:
		pt = pd.Series(index=events.index)

	if (ptS1[1] > 0):
		s1 = -ptS1[1] * events_['trgt']
	else:
		s1 = pd.Series(index=events.index)

	for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
		df0 = close[loc:t1] # path prices
		df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']
		out.loc[loc, 's1'] = df0[df0 < s1[loc]].index.min() # earliest stop loss
		out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min() # earliest profit taking

	return out


def getEvents(close, tEvents, ptS1, trgt, minRet, numThreads, t1=False):

	# Get target
	trg = trgt.loc[tEvents]
	trgt = trgt[trgt > minRet]

	# Get t1 (max holding period)
	if (t1 == False):
		t1 = pd.Series(pd.NaT, index=tEvents)

	# Form events object, apply stop loss on t1
	side_ = pd.Series(1., index=trgt.index)
	events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])

	df0 = mpPandasObject(func=applyPtS1OnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close, events=events, ptS1=[ptS1, ptS1])

	events['t1'] = df0.dropna(how='all').min(axis=1)
	events = events.drop('side', axis=1)
	return events


