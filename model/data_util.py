"""
Kevin Patel
"""
import sys
import os
from functools import partial
from itertools import product
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute

from common_util import is_type, identity_fn, compose, dcompose, pd_dti_index_to_date, filter_cols_below, reindex_on_time_mask, df_downsample_transpose, ser_shift, pd_common_index_rows, chained_filter
from model.common import EXPECTED_NUM_HOURS
from recon.dataset_util import gen_group
from mutate.label_util import prep_labels


""" ********** DATA PREPARATION ********** """
def align_first_last_cols(df, ratio_max=.25, min_num_cols=EXPECTED_NUM_HOURS, how='last'):
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
		alg_df = df.copy(deep=True) 	# Copy made to prevent side-effects in original reference
		cnt_df = alg_df.count()
		first_col, last_col = cnt_df.index[0], cnt_df.index[-1]
		first_valid = alg_df[~alg_df[first_col].isnull() & alg_df[last_col].isnull()]		# Rows where first is valid and last is null
		last_valid = alg_df[alg_df[first_col].isnull() & ~alg_df[last_col].isnull()]		# Rows where last is valid and first is null

		if (how == 'max'):		# The most common occurrence is used
			keep_first = first_valid.size > last_valid.size
		elif (how == 'last'): 	# The last occurrence is used
			if (is_type(alg_df.index, pd.core.index.MultiIndex)):
				fval, lval = first_valid.index[-1][0], last_valid.index[-1][0]
			else:
				fval, lval = first_valid.index[-1], last_valid.index[-1]
			keep_first = fval > lval

		if (keep_first):
			alg_df.loc[alg_df[first_col].isnull() & ~alg_df[last_col].isnull(), :] = last_valid.shift(periods=-1, axis=1)
		else:
			alg_df.loc[~alg_df[first_col].isnull() & alg_df[last_col].isnull(), :] = first_valid.shift(periods=1, axis=1)

		return filter_cols_below(alg_df)
	else:
		return df

def prune_nulls(df, method='ffill', limit=EXPECTED_NUM_HOURS//2):
	if (method=='ffill'):
		return df.dropna(axis=0, how='all').fillna(axis=1, method='ffill', limit=limit).dropna(axis=0, how='any')
	elif (method=='drop'):
		return df.dropna(axis=0, how='any')

def prepare_transpose_data_d(feature_df, row_masks_df):
	"""
	Return delayed object to produce intraday to daily transposed data.
	Converts an intraday single column DataFrame into a daily multi column DataFrame.

	Args:
		feature_df (pd.DataFrame): one series intraday dataframe
	"""
	prep_fn = dcompose(reindex_on_time_mask, df_downsample_transpose, filter_cols_below,
		align_first_last, prune_nulls, partial(pd_dti_index_to_date, new_tz=None))
	return prep_fn(feature_df, row_masks_df)

def prepare_transpose_data(feature_df, row_masks_df):
	"""
	Converts a single indexed intraday DataFrame into a MultiIndexed daily DataFrame.

	Args:
		feature_df (pd.DataFrame): intraday DataFrame
		row_masks_df (pd.DataFrame): row_mask / time mask DataFrame
	"""
	prep_fn = compose(
						reindex_on_time_mask,		# Converts the UTC time index to local time
						df_downsample_transpose,	# Performs the grouby downsample to daily frequency and intraday transpose
						filter_cols_below,			# Filters out columns with 90% or less of their data missing (relative to the most populated column)
						align_first_last_cols,		# Removes an extra column due to misalignment if it exists
						prune_nulls,				# Removes or fills any last null data
						partial(pd_dti_index_to_date, new_tz=None)
					)
	return prep_fn(feature_df, row_masks_df)

def prepare_label_data(label_ser):
	prep_fn = compose(partial(pd_dti_index_to_date, new_tz=None), partial(ser_shift, cast_type=int))
	return prep_fn(label_ser)

def prepare_target_data(label_ser):
	prep_fn = compose(partial(pd_dti_index_to_date, new_tz=None), ser_shift)
	return prep_fn(label_ser)

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
def datagen(dataset, feat_prep_fn=identity_fn, label_prep_fn=identity_fn, target_prep_fn=identity_fn, common_prep_fn=pd_common_index_rows, how='ser_to_ser'):
	"""
	Yield from data generation function.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature df/series
		label_prep_fn: final transform to run on the label df/series
		target_prep_fn: final transform to run on the target df/series
		common_prep_fn: common transform to run after all specific transforms
		how ('ser_to_ser' | 'df_to_ser' | 'ser_to_df'): how to serve the data

	"""
	datagen_fn = {
		'ser_to_ser': datagen_ser_to_ser,
		'df_to_ser': datagen_df_to_ser,
		'df_to_df': datagen_df_to_df
	}.get(how)

	yield from datagen_fn(dataset, feat_prep_fn, label_prep_fn, target_prep_fn, common_prep_fn)

def datagen_ser_to_ser(dataset, feat_prep_fn, label_prep_fn, target_prep_fn, common_prep_fn):
	"""
	Yield data from the dataset by series product.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature series
		label_prep_fn: final transform to run on the label series
		target_prep_fn: final transform to run on the target series

	"""
	for paths, recs, dfs in gen_group(dataset, out=['recs', 'dfs']):
		fpath, lpath, tpath, rpath = paths
		frec, lrec, trec, rrec = recs
		feat_df, lab_df, tar_df, rm_df = (df.compute() for df in dfs)

		logging.debug('fpath: {}'.format(str(fpath)))
		logging.debug('lpath: {}'.format(str(lpath)))
		logging.debug('tpath: {}'.format(str(tpath)))
		logging.debug('rpath: {}'.format(str(rpath)))

		for fcol, lcol, tcol in product(feat_df.columns, lab_df.columns, tar_df.columns):
			feature = feat_prep_fn(feat_df.loc[:, [fcol]], rm_df).dropna(axis=0, how='all')
			label = label_prep_fn(lab_df.loc[:, lcol]).dropna(axis=0, how='all')
			target = target_prep_fn(tar_df.loc[:, tcol]).dropna(axis=0, how='all')
			
			yield (fpath, lpath, tpath, frec, lrec, trec, fcol, lcol, tcol, *common_prep_fn(feature, label, target))

def datagen_df_to_ser(dataset, feat_prep_fn, label_prep_fn, target_prep_fn, common_prep_fn):
	"""
	Yield data from the dataset mapping feature df to each label column.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature df
		label_prep_fn: final transform to run on the label series
		target_prep_fn: final transform to run on the target series

	"""
	for paths, recs, dfs in gen_group(dataset, out=['recs', 'dfs']):
		fpath, lpath, tpath, rpath = paths
		frec, lrec, trec, rrec = recs
		feat_df, lab_df, tar_df, rm_df = (df.compute() for df in dfs)

		logging.debug('fpath: {}'.format(str(fpath)))
		logging.debug('lpath: {}'.format(str(lpath)))
		logging.debug('tpath: {}'.format(str(tpath)))
		logging.debug('rpath: {}'.format(str(rpath)))

		feature = feat_prep_fn(feat_df, rm_df).dropna(axis=0, how='all')

		for lcol, tcol in product(lab_df.columns, tar_df.columns):
			label = label_prep_fn(lab_df.loc[:, lcol]).dropna(axis=0, how='all')
			target = target_prep_fn(tar_df.loc[:, tcol]).dropna(axis=0, how='all')
			
			yield (fpath, lpath, tpath, frec, lrec, trec, lcol, tcol, *common_prep_fn(feature, label, target))

def datagen_df_to_df(dataset, feat_prep_fn, label_prep_fn, target_prep_fn, common_prep_fn):
	"""
	Yield data from the dataset by df product.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: final transform to run on the feature df
		label_prep_fn: final transform to run on the label df
		target_prep_fn: final transform to run on the target df

	"""
	for paths, recs, dfs in gen_group(dataset, out=['recs', 'dfs']):
		fpath, lpath, tpath, rpath = paths
		frec, lrec, trec, rrec = recs
		feat_df, lab_df, tar_df, rm_df = (df.compute() for df in dfs)

		logging.debug('fpath: {}'.format(str(fpath)))
		logging.debug('lpath: {}'.format(str(lpath)))
		logging.debug('tpath: {}'.format(str(tpath)))
		logging.debug('rpath: {}'.format(str(rpath)))

		feature = feat_prep_fn(feat_df, rm_df).dropna(axis=0, how='all')
		label = label_prep_fn(lab_df).dropna(axis=0, how='all')
		target = label_prep_fn(tar_df).dropna(axis=0, how='all')
		
		yield (fpath, lpath, tpath, frec, lrec, trec, *common_prep_fn(feature, label, target))

def hyperopt_trials_to_df(trials):
	"""
	Convert a hyperopt trials object to a pandas DataFrame and return it.
	"""
	pass
