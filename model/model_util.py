# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import filter_cols_below
from model.common import EXPECTED_NUM_HOURS


def align_first_last(df, ratio_max=.25):
	"""
	Return df where non-overlapping subsets have first or last column set to null, align them and remove the redundant column.

	Args:
		df (pd.DataFrame): dataframe of multiple columns
		ratio_max (float): multiplier of the maximum count of all columns, whose product is used as a threshold for the alignment condition.

	Returns:
		Aligned and filtered dataframe if alignment is needed, otherwise return the original dataframe.
	"""
	def fl_alignment_needed(df, ratio_max=ratio_max):
		count_df = df.count()
		return count_df.size > EXPECTED_NUM_HOURS and abs(count_df.iloc[0] - count_df.iloc[-1]) > ratio_max*count_df.max()

	if (fl_alignment_needed(df)):
		cnt_df = df.count()
		first_hr, last_hr = cnt_df.index[0], cnt_df.index[-1]
		firstnull = df[df[first_hr].isnull() & ~df[last_hr].isnull()]
		lastnull = df[~df[first_hr].isnull() & df[last_hr].isnull()]

		# The older format is changed to match the temporally latest one
		if (firstnull.index[-1] > lastnull.index[-1]): 		# Changed lastnull subset to firstnull
			df.loc[~df[first_hr].isnull() & df[last_hr].isnull(), :] = lastnull.shift(periods=1, axis=1)
		elif (firstnull.index[-1] < lastnull.index[-1]):	# Changed firstnull subset to lastnull
			df.loc[df[first_hr].isnull() & ~df[last_hr].isnull(), :] = firstnull.shift(periods=-1, axis=1)

		return filter_cols_below(df)
	else:
		return df

def prune_nulls(df, method='ffill'):
	if (method=='ffill'):
		return df.dropna(axis=0, how='all').fillna(axis=1, method='ffill', limit=3).dropna(axis=0, how='any')
	elif (method=='drop'):
		return df.dropna(axis=0, how='any')
