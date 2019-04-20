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

from common_util import ALL_COLS, is_type, identity_fn, load_json, compose, dcompose, pd_idx_rename, pd_dti_idx_date_only, filter_cols_below, reindex_on_time_mask, df_downsample_transpose, pd_single_ser, ser_shift, pd_common_idx_rows, chained_filter
from model.common import EXPECTED_NUM_HOURS, XG_DIR, DATASET_DIR
from recon.dataset_util import GEN_GROUP_CONSTRAINTS, no_constraint, prep_dataset, gen_group
from mutate.label_util import prep_labels


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
		elif (how == 'last'): 	# The last occurrence is used
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
	if (method=='ffill'):
		return df.dropna(axis=0, how='all').fillna(axis=1, method='ffill', limit=limit).dropna(axis=0, how='any')
	elif (method=='drop'):
		return df.dropna(axis=0, how='any')


""" ********** DATA PREPARATION COMPOSITIONS ********** """
def prepare_transpose_data(feature_df, row_masks_df, delayed=False):
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
				filter_cols_below,			# Filters out columns with 90% or less of their data missing (relative to the most populated column)
				align_first_last_cols,		# Removes an extra column due to misalignment if it exists
				prune_nulls,				# Removes or fills any last null data
				pd_dti_idx_date_only		# Removes the time component of the DatetimeIndex index 
			)
	prep_fn = dcompose(*preproc) if (delayed) else compose(*preproc)
	return prep_fn(feature_df, row_masks_df)

def prepare_label_data(label_ser, delayed=False):
	"""
	Converts a single indexed intraday DataFrame into a MultiIndexed daily DataFrame.

	Args:
		label_ser (pd.Series or pd.DataFrame): Label Series or single column DataFrame
		delayed (boolean): Whether or not to create a delayed function composition

	Returns:
		pd.Series or dask Delayed object
	"""
	preproc = (
				pd_single_ser,							# Makes sure label_ser is a series or single column DataFrame
				pd_dti_idx_date_only,					# Removes the time component of the DatetimeIndex index 
				partial(ser_shift, cast_type=int),		# Shifts the series up by one slot and casts to int
				pd_idx_rename							# Sets the index name to the default
			)
	prep_fn = dcompose(*preproc) if (delayed) else compose(*preproc)
	return prep_fn(label_ser)

def prepare_target_data(target_ser, delayed=False):
	"""
	Converts a single indexed intraday DataFrame into a MultiIndexed daily DataFrame.

	Args:
		target_ser (pd.Series or pd.DataFrame): Target Series or single column DataFrame
		delayed (boolean): Whether or not to create a delayed function composition

	Returns:
		pd.Series or dask Delayed object
	"""
	preproc = (
				pd_single_ser,				# Makes sure target_ser is a series or single column DataFrame
				pd_dti_idx_date_only,		# Removes the time component of the DatetimeIndex index 
				ser_shift,					# Shifts the series up by one slot
				pd_idx_rename				# Sets the index name to the default
			)
	prep_fn = dcompose(*preproc) if (delayed) else compose(*preproc)
	return prep_fn(target_ser)

def prepare_masked_labels(labels_df, label_types, label_filter):
	"""
	XXX - Deprecated
	Return label masked and column filtered dataframe of label series.
	"""
	prepped_labels = prep_labels(labels_df, types=label_types)
	filtered_labels = delayed(lambda df: df.loc[:, chained_filter(df.columns, label_filter)])(prepped_labels) # EOD, FBEOD, FB
	timezone_fixed = delayed(pd_dti_index_to_date)(filtered_labels, new_tz=None)

	return timezone_fixed

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


""" ********** DATA PREP TRANSLATORS ********** """
COMMON_PREP_FNS = {
	'pd_common_idx_rows': pd_common_idx_rows,
}

DATA_PREP_FNS = {
	'prep_transpose_data': prepare_transpose_data,
	'prep_label_data': prepare_label_data,
	'prep_target_data': prepare_target_data,
	'pd_idx_date_only': single_prep_fn(pd_dti_idx_date_only)
}


""" ********** DATA GENERATORS ********** """
def xgdg(xg_fname, delayed=False, **kwargs):
	"""
	Experiment Group Data Generator
	Yields data over an experiment group.

	Args:
		xg_fname (str): experiment group filename
		delayed (bool): whether or not to delay computation, only relevant if 'how' is set to 'df_to_df'
		kwargs: arguments to pass to prep_dataset

	Yields from:
		Generator of tuples of metadata/data
	"""
	xg_dict = load_json(xg_fname, dir_path=XG_DIR)
	dataset_dict = load_json(xg_dict['dataset'], dir_path=DATASET_DIR)
	dataset = prep_dataset(dataset_dict, **kwargs)
	constraint_fn = GEN_GROUP_CONSTRAINTS.get(xg_dict['constraint'], no_constraint)
	common_prep_fn = COMMON_PREP_FNS.get(xg_dict['prep_fn'].get('common', None), pd_common_idx_rows)

	for paths, recs, dfs in gen_group(dataset, group=xg_dict['parts'], out=['recs', 'dfs'], constraint=constraint_fn):
		dpaths = {part: path for part, path in zip(xg_dict['parts'], paths)}
		drecs = {part: rec for part, rec in zip(xg_dict['parts'], recs)}
		ddfs = {part: df for part, df in zip(xg_dict['parts'], dfs)}
		opaths = list(map(lambda x: dpaths[x[0]], xg_dict['how']))
		orecs = list(map(lambda x: drecs[x[0]], xg_dict['how']))

		transformed = []
		for parts in xg_dict['how']:
			prep_fn = DATA_PREP_FNS.get(xg_dict['prep_fn'].get(parts[0], None), identity_fn)
			data = map(lambda x: ddfs[x], parts) if (delayed) else map(lambda x: ddfs[x].compute(), parts)
			transformed.append(prep_fn(*data, delayed=delayed).dropna(axis=0, how='all'))

		if (delayed):
			yield (opaths, orecs, dcompose(common_prep_fn)(*transformed))
		else:
			yield (opaths, orecs, common_prep_fn(*transformed))

def datagen(dataset, feat_prep_fn=identity_fn, label_prep_fn=identity_fn, target_prep_fn=identity_fn, common_prep_fn=pd_common_idx_rows, how='df_to_ser', delayed=False):
	"""
	Yield from data generation function.
	Multiplexes different methods of iterating through the dataset.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: transform to run on the feature df/series
		label_prep_fn: transform to run on the label df/series
		target_prep_fn: transform to run on the target df/series
		common_prep_fn: final function to run on all the data at the end (oftentimes an index intersection function)
		how ('ser_to_ser' | 'df_to_ser' | 'df_to_df'): how to serve the data
		delayed (bool): whether or not to delay computation, only relevant if 'how' is set to 'df_to_df'

	Yields from:
		Generator of tuples of metadata/data
	"""
	datagen_fn = {
		'ser_to_ser': datagen_ser_to_ser,
		'df_to_ser': datagen_df_to_ser,
		'df_to_df': partial(datagen_df_to_df, delayed=delayed)
	}.get(how)

	yield from datagen_fn(dataset, feat_prep_fn, label_prep_fn, target_prep_fn, common_prep_fn)

def datagen_ser_to_ser(dataset, feat_prep_fn, label_prep_fn, target_prep_fn, common_prep_fn):
	"""
	Yield data from the dataset by cartesian product of all feature to label/target series.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: transform to run on the feature series
		label_prep_fn: transform to run on the label series
		target_prep_fn: transform to run on the target series
		common_prep_fn: final function to run on all the data at the end (oftentimes an index intersection function)

	Yields:
		Tuples of metadata/data
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
	Yield data from the dataset by cartesian product of all feature dfs to label/target series.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: transform to run on the feature df
		label_prep_fn: transform to run on the label series
		target_prep_fn: transform to run on the target series
		common_prep_fn: final function to run on all the data at the end (oftentimes an index intersection function)

	Yields:
		Tuples of metadata/data
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
			
			yield (fpath, lpath, tpath, frec, lrec, trec, ALL_COLS, lcol, tcol, *common_prep_fn(feature, label, target))

def datagen_df_to_df(dataset, feat_prep_fn, label_prep_fn, target_prep_fn, common_prep_fn, delayed=False):
	"""
	Yield data from the dataset by cartesian product of all feature dfs to label/target dfs.

	Args:
		dataset: a dataset (recursive dictionary of data returned by prep_dataset)
		feat_prep_fn: transform to run on the feature df
		label_prep_fn: transform to run on the label df
		target_prep_fn: transform to run on the target df
		common_prep_fn: final function to run on all the data at the end (oftentimes an index intersection function)
		delayed (bool): whether or not to delay computation

	Yields:
		Tuples of metadata/data or metadata/delayed
	"""
	for paths, recs, dfs in gen_group(dataset, out=['recs', 'dfs']):
		fpath, lpath, tpath, rpath = paths
		frec, lrec, trec, rrec = recs
		feat_df, lab_df, tar_df, rm_df = dfs if (delayed) else (df.compute() for df in dfs)

		logging.debug('fpath: {}'.format(str(fpath)))
		logging.debug('lpath: {}'.format(str(lpath)))
		logging.debug('tpath: {}'.format(str(tpath)))
		logging.debug('rpath: {}'.format(str(rpath)))

		feature = feat_prep_fn(feat_df, rm_df, delayed=delayed).dropna(axis=0, how='all')
		label = label_prep_fn(lab_df, delayed=delayed).dropna(axis=0, how='all')
		target = target_prep_fn(tar_df, delayed=delayed).dropna(axis=0, how='all')
		
		if (delayed):
			yield (fpath, lpath, tpath, frec, lrec, trec, ALL_COLS, ALL_COLS, ALL_COLS, dcompose(common_prep_fn)(feature, label, target))
		else:
			yield (fpath, lpath, tpath, frec, lrec, trec, ALL_COLS, ALL_COLS, ALL_COLS, *common_prep_fn(feature, label, target))

def hyperopt_trials_to_df(trials):
	"""
	Convert a hyperopt trials object to a pandas DataFrame and return it.
	"""
	pass
