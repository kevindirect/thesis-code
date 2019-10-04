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
from dask import compute

from common_util import ALL_COLS, identity_fn, load_json, compose, dcompose, pd_common_idx_rows
from model.common import XG_DIR, DATASET_DIR
from model.dataprep_util import COMMON_PREP_MAPPING, DATA_PREP_MAPPING
from recon.dataset_util import prep_dataset, gen_group


""" ********** DATA GENERATORS ********** """
def process_group(dataset, group, prep=None, constraint=None, delayed=False):
	"""
	Yields processed dataframes based on provided group, prep function, and path constraint function.

	Args:
		dataset (dict): a loaded dataset
		group (list): parts to traverse in parallel
		prep (str): string pointing to prep function in DATA_PREP_MAPPING with function signature that matches the parts argument
		constraint (str): constraint to match parts by
		delayed (bool): whether or not operations are delayed

	Yields:
		tuple of path, processed data
	"""
	main_part = group[0]
	prep_fn = DATA_PREP_MAPPING.get(prep, identity_fn)

	for paths, recs, dfs in gen_group(dataset, group=group, out=['recs', 'dfs'], constraint=constraint):
		logging.debug(str(paths))
		# Map part->(path, rec, df)
		part_prd = {part: (path, rec, df) for part, path, rec, df in zip(group, paths, recs, dfs)}
		main_path, main_rec = part_prd[main_part][0], part_prd[main_part][1]
		# Get the dataframe(s) in the order of parts
		input_data = map(lambda x: part_prd[x][-1], parts) if (delayed) else map(lambda x: part_prd[x][-1].compute(), group)
		transformed = prep_fn(*input_data, delayed=delayed).dropna(axis=0, how='all')
		yield (main_path, transformed)

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
	common_prep_fn = COMMON_PREP_MAPPING.get(xg_dict['prep_fn'].get('common', None), pd_common_idx_rows)

	for paths, recs, dfs in gen_group(dataset, group=xg_dict['parts'], out=['recs', 'dfs'], constraint=xg_dict['constraint']):
		dpaths = {part: path for part, path in zip(xg_dict['parts'], paths)}
		drecs = {part: rec for part, rec in zip(xg_dict['parts'], recs)}
		ddfs = {part: df for part, df in zip(xg_dict['parts'], dfs)}

		# Path/rec used to represent each output segment is the first in each list
		opaths = list(map(lambda x: dpaths[x[0]], xg_dict['how']))
		orecs = list(map(lambda x: drecs[x[0]], xg_dict['how']))

		transformed = []
		for parts in xg_dict['how']:
			prep_fn = DATA_PREP_MAPPING.get(xg_dict['prep_fn'].get(parts[0], None), identity_fn)
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
