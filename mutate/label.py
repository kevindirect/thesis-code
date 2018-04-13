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

from common_util import DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.ops import *


UP = 1
DOWN = -1
SIDEWAYS = 0

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

def intraday_triple_barrier():
	"""
	Return intraday triple barrier thresholded label series.

	Args:
		ser (pd.Series): series of changes to threshold
		thresh (float or pd.Series, ℝ≥0): float or float series to threshold on
		scalar (float, ℝ≥0): threshold multiplier

	Returns:
		pd.Series ∈ {-1, 0, 1}
	"""


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


