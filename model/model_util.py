# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from dask import delayed

from common_util import df_dti_index_to_date, chained_filter, filter_cols_below, reindex_on_time_mask, gb_transpose
from model.common import EXPECTED_NUM_HOURS
from recon.dataset_util import prep_labels


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

def prepare_transpose_data(features_df, row_masks_df, feat_select_filter):
	"""
	Return delayed object to produce intraday to daily transposed data.
	"""
	reindexed = delayed(reindex_on_time_mask)(features_df, row_masks_df)
	selected = delayed(lambda df: df.loc[:, chained_filter(df.columns, feat_select_filter)])(reindexed)
	transposed = delayed(gb_transpose)(selected)
	filtered = delayed(filter_cols_below)(transposed)
	aligned = delayed(align_first_last)(filtered)
	pruned = delayed(prune_nulls)(aligned)
	timezone_fixed = delayed(df_dti_index_to_date)(pruned, new_tz=None)

	return timezone_fixed

def prepare_masked_labels(labels_df, label_types, label_filter):
	"""
	Return label masked and column filtered dataframe of label series.
	"""
	prepped_labels = prep_labels(labels_df, types=label_types)
	filtered_labels = delayed(lambda df: df.loc[:, chained_filter(df.columns, label_filter)])(prepped_labels) # EOD, FBEOD, FB
	timezone_fixed = delayed(df_dti_index_to_date)(filtered_labels, new_tz=None)

	return timezone_fixed
