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
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adadelta, Adam, Adamax, Nadam
from common_util import RECON_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, get_cmd_args, reindex_on_time_mask, gb_transpose, pd_common_index_rows, filter_cols_below, dump_df, load_json, outer_join, list_get_dict, chained_filter, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, EXPECTED_NUM_HOURS, default_dataset, default_filterset, default_nt_filter, default_target_col_idx
from recon.dataset_util import prep_dataset, prep_labels, gen_group
from recon.model_util import get_train_test_split, gen_time_series_split
from recon.label_util import shift_label


def net_test(argv):
	cmd_arg_list = ['dataset=', 'filterset=', 'idxfilters=', 'assets=', 'target_col_idx=', 'visualize']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='net_test')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	filterset_name = cmd_input['filterset='] if (cmd_input['filterset='] is not None) else default_filterset
	filter_idxs =  list(map(str.strip, cmd_input['idxfilters='].split(','))) if (cmd_input['idxfilters='] is not None) else default_nt_filter
	assets = list(map(str.strip, cmd_input['assets='].split(','))) if (cmd_input['assets='] is not None) else None
	target_col_idx = int(cmd_input['target_col_idx=']) if (cmd_input['target_col_idx='] is not None) else default_target_col_idx
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
			logging.info('fpaths: ' +str(fpaths))
			logging.info('lpaths: ' +str(lpaths))
			logging.info('rpaths: ' +str(rpaths))

			reindexed = delayed(reindex_on_time_mask)(features, row_masks)
			transposed = delayed(gb_transpose)(reindexed.loc[:, ['pba_avgPrice']])
			filtered = delayed(filter_cols_below)(transposed)
			aligned = delayed(align_first_last)(filtered)
			pruned = delayed(prune_nulls)(aligned)

			prepped_labels = prep_labels(labels, types=['bool'])
			filtered_labels = delayed(lambda df: df.loc[:, chained_filter(df.columns, labs_filter)])(prepped_labels) # EOD, FBEOD, FB
			
			ff_test = delayed(feedforward_test)(pruned, filtered_labels, label_col_idx=target_col_idx)
			ff_test.compute()

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

def feedforward_test(feat_df, lab_df, label_col_idx=0):
	lab_name, num_features = lab_df.columns[label_col_idx], feat_df.shape[1]
	features, label = pd_common_index_rows(feat_df.dropna(axis=0, how='all'), shift_label(lab_df.loc[:, lab_name]).dropna())

	# label[label==-1] = 0
	logging.info('DATA DESCRIPTION')
	logging.info('label description: \n{}'.format(label.value_counts(normalize=True, sort=True).to_frame().T))
	logging.info('num features: {}'.format(num_features))

	feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, label, train_ratio=.8)

	opt = SGD(lr=0.0001, momentum=0.00001, decay=0.0, nesterov=False)
	# opt = RMSprop(lr=0.0001, rho=0.99, epsilon=None, decay=0.0)
	model = Sequential()
	model.add(Dense(num_features, input_dim=num_features, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros',
		kernel_regularizer=None, bias_regularizer=None))
	model.add(Dense(1, input_dim=num_features, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	print('BEFORE')
	for idx, layer in enumerate(model.layers):
		print('layer[{idx}] weights: \n{weights}'.format(idx=idx, weights=str(layer.get_weights())))

	model.fit(x=feat_train, y=lab_train, epochs=20, batch_size=128)
	score = model.evaluate(feat_test, lab_test, batch_size=128)
	forecast = model.predict(feat_test, batch_size=128)

	print('AFTER')
	for idx, layer in enumerate(model.layers):
		print('layer[{idx}] weights: \n{weights}'.format(idx=idx, weights=str(layer.get_weights())))
	print('summary: {summary}'.format(summary=str(model.summary())))
	logging.debug('{metrics}: {score}'.format(metrics=str(model.metrics_names), score=str(score)))
	logging.debug('actual: {actual} \nforecast: {forecast}'.format(actual=str(lab_test), forecast=str(forecast.T[0])))

	return score


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		net_test(sys.argv[1:])
