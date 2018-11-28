# Kevin Patel

import sys
import os
from functools import partial
from itertools import product
import logging

import numpy as np
import pandas as pd
from dask import delayed

from common_util import identity_fn, compose, pd_dti_index_to_date, filter_cols_below, reindex_on_time_mask, gb_transpose, ser_shift, chained_filter
from model.common import EXPECTED_NUM_HOURS
from recon.dataset_util import gen_group
from mutate.label_util import prep_labels


""" ********** DATA PREPARATION ********** """
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

def prepare_transpose_data_d(feature_df, row_masks_df):
	"""
	Return delayed object to produce intraday to daily transposed data.
	Converts an intraday single column DataFrame into a daily multi column DataFrame.

	Args:
		feature_df (pd.DataFrame): one series intraday dataframe
	"""
	reindexed = delayed(reindex_on_time_mask)(feature_df, row_masks_df)
	transposed = delayed(gb_transpose)(reindexed)
	filtered = delayed(filter_cols_below)(transposed)
	aligned = delayed(align_first_last)(filtered)
	pruned = delayed(prune_nulls)(aligned)
	timezone_fixed = delayed(pd_dti_index_to_date)(pruned, new_tz=None)

	return timezone_fixed

def prepare_transpose_data(feature_df, row_masks_df):
	"""
	Converts an intraday single column DataFrame into a daily multi column DataFrame.

	Args:
		feature_df (pd.DataFrame): one series intraday dataframe
	"""
	reindexed = reindex_on_time_mask(feature_df, row_masks_df)
	transposed = gb_transpose(reindexed)
	filtered = filter_cols_below(transposed)
	aligned = align_first_last(filtered)
	pruned = prune_nulls(aligned)
	timezone_fixed = pd_dti_index_to_date(pruned, new_tz=None)

	return timezone_fixed

def prepare_label_data(label_ser):
	return compose(partial(pd_dti_index_to_date, new_tz=None), ser_shift)

def prepare_masked_labels(labels_df, label_types, label_filter):
	"""
	XXX - Deprecated
	Return label masked and column filtered dataframe of label series.
	"""
	prepped_labels = prep_labels(labels_df, types=label_types)
	filtered_labels = delayed(lambda df: df.loc[:, chained_filter(df.columns, label_filter)])(prepped_labels) # EOD, FBEOD, FB
	timezone_fixed = delayed(pd_dti_index_to_date)(filtered_labels, new_tz=None)

	return timezone_fixed


""" ********** DATA GENERATORS ********** """
def datagen(dataset, feat_prep_fn=identity_fn, label_prep_fn=identity_fn, how='ser_to_ser'):
	"""
	Yield from data generation function.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature df/series
		label_prep_fn: final transform to run on the label df/series
		how ('ser_to_ser' | 'df_to_ser' | 'ser_to_df')

	"""
	datagen_fn = {
		'ser_to_ser': datagen_ser_to_ser,
		'df_to_ser': datagen_df_to_ser,
		'df_to_df': datagen_df_to_df
	}.get(how)

	yield from datagen_fn(dataset, feat_prep_fn, label_prep_fn)

def datagen_ser_to_ser(dataset, feat_prep_fn, label_prep_fn):
	"""
	Yield data from the dataset by series product.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature df/series
		label_prep_fn: final transform to run on the label df/series

	"""
	for paths, dfs in gen_group(dataset):
		fpaths, lpaths, rpaths = paths
		features, labels, row_masks = (df.compute() for df in dfs)

		logging.debug('fpaths: {}'.format(str(fpaths)))
		logging.debug('lpaths: {}'.format(str(lpaths)))
		logging.debug('rpaths: {}'.format(str(rpaths)))

		for feat_col, label_col in product(features.columns, labels.columns):
			feature = feat_prep_fn(features.loc[:, [feat_col]], row_masks).dropna(axis=0, how='all')
			label = label_prep_fn(labels.loc[:, label_col]).dropna(axis=0, how='all')
			
			yield feature, label

def datagen_df_to_ser(dataset, feat_prep_fn, label_prep_fn):
	"""
	Yield data from the dataset mapping feature df to each label column.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature df/series
		label_prep_fn: final transform to run on the label df/series

	"""
	for paths, dfs in gen_group(dataset):
		fpaths, lpaths, rpaths = paths
		features, labels, row_masks = dfs

		logging.debug('fpaths: {}'.format(str(fpaths)))
		logging.debug('lpaths: {}'.format(str(lpaths)))
		logging.debug('rpaths: {}'.format(str(rpaths)))

		feature = feat_prep_fn(features, row_masks).dropna(axis=0, how='all')

		for label_col in labels.columns:
			label = label_prep_fn(labels.loc[:, label_col]).dropna(axis=0, how='all')
			
			yield feature, label

def datagen_df_to_df(dataset, feat_prep_fn, label_prep_fn):
	"""
	Yield data from the dataset by df product.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature df/series
		label_prep_fn: final transform to run on the label df/series

	"""
	for paths, dfs in gen_group(dataset):
		fpaths, lpaths, rpaths = paths
		features, labels, row_masks = dfs

		logging.debug('fpaths: {}'.format(str(fpaths)))
		logging.debug('lpaths: {}'.format(str(lpaths)))
		logging.debug('rpaths: {}'.format(str(rpaths)))

		feature = feat_prep_fn(features, row_masks).dropna(axis=0, how='all')
		label = label_prep_fn(labels).dropna(axis=0, how='all')
		
		yield feature, label

def hyperopt_trials_to_df(trials):
	"""
	Convert a hyperopt trials object to a pandas DataFrame and return it.
	"""
	pass
