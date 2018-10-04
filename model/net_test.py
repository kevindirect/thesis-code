# Kevin Patel

import sys
import os
from os import sep
from os.path import splitext
from functools import partial, reduce
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute, visualize
from keras.models import Sequential
from keras.layers import Dense, Activation

from common_util import RECON_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, get_cmd_args, reindex_on_time_mask, gb_transpose, filter_cols_below, dump_df, load_json, outer_join, list_get_dict, chained_filter, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, EXPECTED_NUM_HOURS, default_dataset, default_filterset, default_nt_filter
from recon.dataset_util import prep_dataset, prep_labels, gen_group
from recon.model_util import get_train_test_split, gen_time_series_split
from recon.label_util import shift_label


def net_test(argv):
	cmd_arg_list = ['dataset=', 'filterset=', 'idxfilters=', 'assets=', 'visualize']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='net_test')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	filterset_name = cmd_input['filterset='] if (cmd_input['filterset='] is not None) else default_filterset
	filter_idxs =  list(map(str.strip, cmd_input['idxfilters='].split(','))) if (cmd_input['idxfilters='] is not None) else default_nt_filter
	assets = list(map(str.strip, cmd_input['assets='].split(','))) if (cmd_input['assets='] is not None) else None
	run_compute = True if (cmd_input['visualize'] is None) else False

	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	filter_dict = load_json(filterset_name, dir_path=FILTERSET_DIR)

	filterset = []
	for filter_idx in filter_idxs:
		selected = [flt for flt in filter_dict[filter_idx] if (flt not in filterset)]
		filterset.extend(selected)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map={'features': filterset})

	logging.info('assets: ' +str('all' if (assets==None) else ', '.join(assets)))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']['paths']), dataset_name[:-JSON_SFX_LEN]))
	logging.info('filter: {} [{}]'.format(filterset_name[:-JSON_SFX_LEN], str(', '.join(filter_idxs))))
	logging.debug('filterset: ' +str(filterset))
	logging.debug('fpaths: ' +str(dataset['features']['paths']))
	logging.debug('lpaths: ' +str(dataset['labels']['paths']))
	logging.debug('rmpaths: ' +str(dataset['row_masks']['paths']))

	labs_filter = [
	{
		"exact": [],
		"startswith": ["pba_"],
		"endswith": [],
		"regex": [],
		"exclude": None
	},
	{
		"exact": [],
		"startswith": [],
		"endswith": ["_eod", "_fb", "_fbeod"],
		"regex": [],
		"exclude": None
	}]

	final_dfs = {}
	if (run_compute):
		logging.info('executing...')
		for paths, dfs in gen_group(dataset):
			fpaths, lpaths, rpaths = paths
			features, labels, row_masks = dfs
			asset = fpaths[0]
			logging.info(fpaths, lpaths, rpaths)

			reindexed = delayed(reindex_on_time_mask)(features, row_masks)
			transposed = delayed(gb_transpose)(reindexed.loc[:, ['pba_avgPrice']])
			cleaned = delayed(filter_cols_below)(transposed)
			to_align = delayed(is_alignment_needed)(cleaned)
			final_feats = delayed(lambda df, align: df if (not align) else align_first_last(transposed))(cleaned, to_align)

			final_labs = prep_labels(labels, types=['bool'])
			final_labs = delayed(lambda df: df.loc[:, chained_filter(df.columns, labs_filter)])(final_labs) # EOD, FBEOD, FB
			
			sc = delayed(regular_test)(final_feats, final_labs)
			print(sc.compute())


def is_alignment_needed(df, ratio_max=.25):
	count_df = df.count()
	return count_df.size > EXPECTED_NUM_HOURS and abs(count_df.iloc[0] - count_df.iloc[-1]) > ratio_max*count_df.max()

def align_first_last(df):
	"""
	Return df where non-overlapping subsets have first or last column set to null, align them and remove the redundant column.
	"""
	cnt_df = df.count()
	first_hr, last_hr = cnt_df.index[0], cnt_df.index[-1]
	firstnull = df[df[first_hr].isnull() & ~df[last_hr].isnull()]
	lastnull = df[~df[first_hr].isnull() & df[last_hr].isnull()]

	# The older format is changed to match the latest one
	if (firstnull.index[-1] > lastnull.index[-1]): 		# Changed lastnull subset to firstnull
		df.loc[~df[first_hr].isnull() & df[last_hr].isnull(), :] = lastnull.shift(periods=1, axis=1)
	elif (firstnull.index[-1] < lastnull.index[-1]):	# Changed firstnull subset to lastnull
		df.loc[df[first_hr].isnull() & ~df[last_hr].isnull(), :] = firstnull.shift(periods=-1, axis=1)

	return filter_cols_below(df)

def regular_test(feat_df, lab_df):
	lab_ser = shift_label(lab_df.iloc[:, 0])
	model = Sequential()
	num_features = feat_df.shape[1]
	model.add(Dense(num_features*2, input_dim=num_features, activation='sigmoid'))
	model.add(Dense(num_features, input_dim=num_features, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	feat_train, feat_test, lab_train, lab_test = get_train_test_split(feat_df.dropna(axis=0, how='all'), lab_ser.dropna())
	model.fit(feat_train, lab_train, epochs=20, batch_size=128)
	score = model.evaluate(feat_test, lab_test, batch_size=128)

	return score


def cnn_test(dataset):

	for lpath in dataset['labels']['paths']:
		asset_name, base_label_name = lpath[0], lpath[-1]
		test_name = '_'.join([meta['pfx_fmt'].format(**variant), base_label_name])
		logging.info(asset_name +': ' +test_name)

		label_df = list_get_dict(dataset['labels']['dfs'], lpath)
		labels = prep_labels(label_df)

		for fpath in filter(lambda fpath: fpath[0]==asset_name, dataset['features']['paths']):
			feat_df_desc = fpath[-1]
			logging.debug(feat_df_desc)
			feats = list_get_dict(dataset['features']['dfs'], fpath)
			sub_matrix = delayed(gta_apply_type_fn)(feats, feat_df_desc, labels, partial(gta_test_fn, **variant))
			matrix = delayed(pd.concat)([matrix, sub_matrix], axis=0, join='outer', ignore_index=True, sort=False)


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		net_test(sys.argv[1:])
