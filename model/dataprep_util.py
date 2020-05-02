"""
Kevin Patel
"""
import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import is_type, compose, dcompose, pd_idx_rename, pd_idx_to_midx, pd_dti_idx_date_only, filter_cols_below, reindex_on_time_mask, df_downsample_transpose, pd_single_ser, ser_shift, pd_common_idx_rows, df_midx_restack
from model.common import EXPECTED_NUM_HOURS


""" ********** DATA PREPARATION UTIL FUNCTIONS ********** """
def align_first_last_cols(df, ratio_max=.25, min_num_cols=EXPECTED_NUM_HOURS, how='last', deep=False):
	"""
	Return a df where a misalignment problem of data within columns causing a redundant column is fixed.

	If the passed DataFrame of n+1 columns has rows with n valid column values, where some rows are
	column-indexed from [0, n-1] and others are column-indexed from [1, n], then this function
	will shift the less commonly occurring indexing method so that the returned DataFrame has n columns.

	Args:
		df (pd.DataFrame): Dataframe of at least two columns
		ratio_max (float): Threshold value to test for misalignment, alignment is conducted if the difference
			in non-null counts between the first and last column is greater than ratio_max * the most populated column count
		min_num_cols (int): Only perform aligment if there are more than this number of columns
		how (['max'|'last']): How to choose whether to drop the first or last column label if needed
		deep (boolean): Whether to create a deepcopy to avoid side-effects on the passed object

	Returns:
		DataFrame
	"""
	def fl_alignment_needed(df, ratio_max=ratio_max):
		"""
		Returns whether alignment (redundant column removal) is needed.
		"""
		count_df = df.count()
		fl_difference = abs(count_df.iloc[0] - count_df.iloc[-1])
		return fl_difference > ratio_max*count_df.max()

	if (df.columns.size > min_num_cols and fl_alignment_needed(df)):
		if (deep):
			df = df.copy(deep=True)
		cnt_df = df.count()
		first_col, last_col = cnt_df.index[0], cnt_df.index[-1]
		first_valid = df[~df[first_col].isnull() & df[last_col].isnull()]		# Rows where first is valid and last is null
		last_valid = df[df[first_col].isnull() & ~df[last_col].isnull()]		# Rows where last is valid and first is null

		if (how == 'max'):		# The most common occurrence is used
			keep_first = first_valid.size > last_valid.size
		elif (how == 'last'):		# The last occurrence is used
			if (is_type(df.index, pd.core.index.MultiIndex)):
				fval, lval = first_valid.index[-1][0], last_valid.index[-1][0]
			else:
				fval, lval = first_valid.index[-1], last_valid.index[-1]
			keep_first = fval > lval

		if (keep_first):
			df.loc[df[first_col].isnull() & ~df[last_col].isnull(), :] = last_valid.shift(periods=-1, axis=1)
		else:
			df.loc[~df[first_col].isnull() & df[last_col].isnull(), :] = first_valid.shift(periods=1, axis=1)

		return filter_cols_below(df)
	else:
		return df

def prune_nulls(df, method='ffill', limit=EXPECTED_NUM_HOURS//2):
	"""
	Conveninence Function to remove nulls/NaNs from df or series rows.
	"""
	if (method=='ffill'):
		return df.dropna(axis=0, how='all').fillna(axis=1, method='ffill', limit=limit).dropna(axis=0, how='any') # Drops rows that start with null values or exceed the limit
	elif (method=='drop'):
		return df.dropna(axis=0, how='any')

def single_prep_fn(fn):
	"""
	Wraps a single function into a delayable prepreocessing function

	Args:
		fn: function that takes in data and returns data

	Returns:
		function with signature "function(object, delayed)"
	"""
	def prep_fn(pd_obj, delayed=False):
		preproc_fn = dcompose(fn) if (delayed) else fn
		return preproc_fn(pd_obj)
	return prep_fn


""" ********** DATA PREPARATION COMPOSITIONS ********** """
def prep_transpose_data(feature_df, row_masks_df, delayed=False):
	"""
	Converts a single indexed intraday DataFrame into a MultiIndexed daily DataFrame.

	Args:
		feature_df (pd.DataFrame): Intraday DataFrame
		row_masks_df (pd.DataFrame): DataFrame of row masks / time mask
		delayed (boolean): Whether or not to create a delayed function composition

	Returns:
		pd.DataFrame or dask Delayed object
	"""
	preproc = (
				reindex_on_time_mask,		# Converts the UTC time index to local time
				df_downsample_transpose,	# Performs the grouby downsample to daily frequency and intraday transpose
				filter_cols_below,		# Filters out columns with 90% or less of their data missing (relative to the most populated column)
				align_first_last_cols,		# Removes an extra column due to misalignment if it exists
				prune_nulls,			# Removes or fills any last null data
				pd_dti_idx_date_only,		# Removes the time component of the DatetimeIndex index
				df_midx_restack			# Restacks to fix https://github.com/pandas-dev/pandas/issues/2770
			)
	prep_fn = dcompose(*preproc) if (delayed) else compose(*preproc)
	return prep_fn(feature_df, row_masks_df)


def prep_stack_data(feature_df, delayed=False):
	"""
	Converts a single index daily DataFrame into a single column MultiIndex daily DataFrame.

	Args:
		feature_df (pd.DataFrame): Daily DataFrame
		delayed (boolean): Whether or not to create a delayed function composition

	Returns:
		pd.DataFrame or dask Delayed object
	"""
	preproc = (
				pd_dti_idx_date_only,			# Removes the time component of the DatetimeIndex index
				partial(pd_idx_to_midx, col_name=-1)	# Converts to MultiIndex DF
			)
	prep_fn = dcompose(*preproc) if (delayed) else compose(*preproc)
	return prep_fn(feature_df)


def prep_label_data(label_ser, delayed=False):
	"""
	Prepares a label series (includes time shift).

	Args:
		label_ser (pd.Series or pd.DataFrame): Label Series or single column DataFrame
		delayed (boolean): Whether or not to create a delayed function composition

	Returns:
		pd.Series or dask Delayed object
	"""
	preproc = (
				pd_single_ser,					# Makes sure label_ser is a series or single column DataFrame
				pd_dti_idx_date_only,				# Removes the time component of the DatetimeIndex index
				partial(ser_shift, cast_type=int),		# Shifts the series up by one slot and casts to int
				pd_idx_rename					# Sets the index name to the default
			)
	prep_fn = dcompose(*preproc) if (delayed) else compose(*preproc)
	return prep_fn(label_ser)

def prep_target_data(target_ser, delayed=False):
	"""
	Prepares a target series (includes time shift).

	Args:
		target_ser (pd.Series or pd.DataFrame): Target Series or single column DataFrame
		delayed (boolean): Whether or not to create a delayed function composition

	Returns:
		pd.Series or dask Delayed object
	"""
	preproc = (
				pd_single_ser,			# Makes sure target_ser is a series or single column DataFrame
				pd_dti_idx_date_only,		# Removes the time component of the DatetimeIndex index
				ser_shift,			# Shifts the series up by one slot
				pd_idx_rename			# Sets the index name to the default
			)
	prep_fn = dcompose(*preproc) if (delayed) else compose(*preproc)
	return prep_fn(target_ser)


""" ********** DATA PREP MAPPINGS ********** """
COMMON_PREP_MAPPING = {
	'pd_common_idx_rows': pd_common_idx_rows,
}

DATA_PREP_MAPPING = {
	'prep_transpose_data': prep_transpose_data,
	'prep_stack_data': prep_stack_data,
	'prep_label_data': prep_label_data,
	'prep_target_data': prep_target_data,
	'pd_idx_date_only': single_prep_fn(pd_dti_idx_date_only)
}
