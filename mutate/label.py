# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize, float64

from common_util import DT_BIZ_DAILY_FREQ, inner_join, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum

LABEL_SFX_LEN = len('_dir')
UP = 1
DOWN = -1
SIDEWAYS = 0

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
# @vectorize([float64(float64, float64, float64)], nopython=True)
# def thresh_break(start, end, thresh):
# 	change = (end / start) - 1

# 	return change if (change > thresh or change < -thresh) else 0

# def find_touch(group_df, per_shift=1):
# 	"""
# 	Return touch found

# 	Args:
# 		group_df (pd.DataFrame): dataframe of aggregation period
# 		per_shift (integer, ℤ): number to shift break period by
# 	"""
# 	group_df = group_df.dropna()
# 	if (group_df.empty):
# 		return np.NaN

# 	start_arr = np.array(group_df.loc[:, 'start'].first(DT_HOURLY_FREQ))
# 	end_arr = np.array(group_df['end'].values)
# 	thresh_arr = np.array(group_df['thresh'].values)

# 	stats = {
# 		"dir": 0,
# 		"mag": 0,
# 		"brk": 0,
# 		"day": end_arr.size
# 	}

# 	breaks = thresh_break(start_arr, end_arr, thresh_arr)
# 	break_ids = np.flatnonzero(breaks)
	
# 	if (break_ids.size != 0):
# 		# Change set to first threshold break
# 		change = breaks[break_ids[0]]
# 		stats['brk'] = break_ids[0] + per_shift
# 	else:
# 		# End of day change, no threshold
# 		change = (end_arr[-1] / start_arr[0]) - 1

# 	stats['dir'] = np.sign(change)
# 	stats['mag'] = abs(change)

# 	return stats['dir'], stats['mag'], stats['brk'], stats['day']

# def intraday_triple_barrier(intraday_df, thresh, up_scalar=1.0, down_scalar=1.0, agg_freq=DT_BIZ_DAILY_FREQ):
# 	"""
# 	Return intraday triple barrier label series.
	
# 	Args:
# 		intraday_df (pd.DataFrame): intraday price dataframe
# 		thresh (String): name of threshold column
# 		scalar (dict(str: float), ℝ≥0): bull/bear thresh multipliers
	
# 	Returns:
# 		Return pd.DataFrame with four columns:
# 			- 'dir': price direction
# 			- 'spd': change speed
# 			- 'brk': period of break (zero if none)
# 			- 'day': number of trading periods
# 	"""
# 	# DF Preprocessing
# 	col_renames = {
# 		intraday_df.columns[0]: "start",
# 		thresh: "thresh"
# 	}
# 	num_cols = len(intraday_df.columns)
# 	if (num_cols == 2):
# 		intraday_df['end'] = intraday_df[intraday_df.columns[0]]
# 	elif (num_cols > 2):
# 		col_renames[intraday_df.columns[1]] = 'end'
# 	intraday_df.rename(columns=col_renames, inplace=True)
	
# 	# Threshold scale
# 	intraday_df['thresh'] = scalar * intraday_df['thresh']

# 	# Apply
# 	labels = intraday_df.groupby(pd.Grouper(freq=agg_freq)).apply(find_touch)
# 	labels = labels.apply(pd.Series)
# 	labels.columns=['dir','mag', 'brk', 'day']
# 	return labels


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def thresh_break(change, thresh, up_scalar, down_scalar):
	up_thresh = up_scalar * thresh
	down_thresh = down_scalar * thresh

	return change if (change >= up_thresh or change <= -down_thresh) else 0


def triple_barrier_label(group_df, up_scalar, down_scalar, shift_comp=1):
	group_df = group_df.dropna() # don't perform inplace ops in a gb-apply function
	if (group_df.empty):
		return None

	ret_arr = np.array(group_df[group_df.columns[0]].values)
	thresh_arr = np.array(group_df[group_df.columns[1]].values)

	stats = {
		"dir": 0,
		"mag": 0,
		"brk": 0,
		"nmb": 0,
		"nmt": 0
		# "day": ret_arr.size
	}

	breaks = thresh_break(ret_arr, thresh_arr, up_scalar, down_scalar)
	break_ids = np.flatnonzero(breaks)
	
	if (break_ids.size > 0):
		# Change set to first threshold break
		change = breaks[break_ids[0]]
		stats['brk'] = break_ids[0] + shift_comp
	else:
		# End of day change, no threshold
		change = ret_arr[-1]

	stats['dir'] = np.sign(change)										# change sign
	stats['mag'] = abs(change)											# change magnitude
	stats['nmb'] = np.count_nonzero(np.sign(breaks) == np.sign(change))	# number breaks in the labelled direction
	stats['nmt'] = break_ids.size										# number of total breaks

	# XXX - lmn: largest motonic subsequence
	# XXX - smn: slope of largest monotonic subsequence

	return pd.DataFrame([stats], index=[int(group_df.index[-1].hour)])


def intraday_triple_barrier(intraday_df, scalar=(1.0, 1.0), agg_freq=DT_BIZ_DAILY_FREQ):
	"""
	Return intraday triple barrier label stats.
	
	Args:
		intraday_df (pd.DataFrame): intraday price dataframe
		scalar (tuple(float, float)): determines the scale factor on the upper and lower barriers
	
	Returns:
		Return pd.DataFrame
	"""
	label_df = intraday_df.groupby(pd.Grouper(freq=agg_freq)).apply(triple_barrier_label, up_scalar=abs(scalar[0]), down_scalar=abs(scalar[1]))

	# Fix multi index: (date, hour) -> datetime
	label_df.index = label_df.index.map(lambda x: x[0].replace(hour=x[1]))

	# if (drop_the_base):
	# 	label_df.drop(intraday_df.columns, axis=1, inplace=True)
	# 	label_df = label_df[['dir', 'mag', 'brk', 'nmb', 'nmt']]

	return label_df[['dir', 'mag', 'brk', 'nmb', 'nmt']]






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


