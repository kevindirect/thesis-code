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
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import SGD, RMSprop, Adadelta, Adam, Adamax, Nadam

from common_util import RECON_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, get_cmd_args, in_debug_mode, pd_common_index_rows, dump_df, load_json, outer_join, list_get_dict, chained_filter, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, default_dataset, default_filterset, default_nt_filter, default_target_col_idx
from model.model_util import prepare_transpose_data, prepare_masked_labels
from recon.dataset_util import prep_dataset, prep_labels, gen_group
from recon.split_util import get_train_test_split, gen_time_series_split
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

	feats_filter = [{
		"exact": ["pba_avgPrice"],
		"startswith": [],
		"endswith": [],
		"regex": [],
		"exclude": None
	}]

	labs_filter = [ # EOD, FBEOD, FB
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
		"endswith": ["_eod(0%)", "_eod(1%)", "_eod(2%)", "_fb", "_fbeod"],
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

			final_feats = prepare_transpose_data(features, row_masks, feats_filter)
			final_labels = prepare_masked_labels(labels, ['bool'], labs_filter)
			print('final_feats', final_feats.compute().dropna(axis=0, how='all'))
			print('final_feats idx', final_feats.compute().dropna(axis=0, how='all').index)
			print('final_labels', shift_label(final_labels.compute().iloc[:, 0]))
			print('final_labels idx', shift_label(final_labels.compute().iloc[:, 0]).index)
			ff_test = delayed(feedforward_test)(final_feats, final_labels, label_col_idx=target_col_idx)
			ff_test.compute()


def feedforward_test(feat_df, lab_df, label_col_idx=0):
	lab_name, num_features = lab_df.columns[label_col_idx], feat_df.shape[1]
	features, label = pd_common_index_rows(feat_df.dropna(axis=0, how='all'), shift_label(lab_df.loc[:, lab_name]).dropna())

	# label[label==-1] = 0
	logging.info('DATA DESCRIPTION')
	logging.info('label[{name}]: \n{vc}'.format(name=lab_name, vc=label.value_counts(normalize=True, sort=True).to_frame().T))
	logging.info('num features: {}'.format(num_features))

	feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, label, train_ratio=.8)

	opt = SGD(lr=0.0001, momentum=0.00001, decay=0.0, nesterov=False)
	# opt = RMSprop(lr=0.0001, rho=0.99, epsilon=None, decay=0.0)
	model = Sequential()
	model.add(Dense(num_features*2, input_dim=num_features, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros',
		kernel_regularizer=None, bias_regularizer=None))
	model.add(Dense(num_features*2, input_dim=num_features*2, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros',
		kernel_regularizer=None, bias_regularizer=None))
	model.add(Dropout(.2))
	model.add(Dense(num_features, input_dim=num_features*2, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros',
		kernel_regularizer=None, bias_regularizer=None))
	model.add(Dense(1, input_dim=num_features, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	logging.info('BEFORE')
	if (in_debug_mode()):
		for idx, layer in enumerate(model.layers):
			logging.debug('layer[{idx}] weights: \n{weights}'.format(idx=idx, weights=str(layer.get_weights())))

	model.fit(x=feat_train, y=lab_train, epochs=50, batch_size=128)
	score = model.evaluate(feat_test, lab_test, batch_size=128)

	logging.info('AFTER')
	print('summary: {summary}'.format(summary=str(model.summary())))
	print('{metrics}: {score}'.format(metrics=str(model.metrics_names), score=str(score)))

	if (in_debug_mode()):
		for idx, layer in enumerate(model.layers):
			logging.debug('layer[{idx}] weights: \n{weights}'.format(idx=idx, weights=str(layer.get_weights())))
		forecast = model.predict(feat_test, batch_size=128)
		logging.debug('actual: {actual} \nforecast: {forecast}'.format(actual=str(lab_test), forecast=str(forecast.T[0])))

	return score


def rnn_test(feat_df, lab_df, label_col_idx=0):
	lab_name, num_features = lab_df.columns[label_col_idx], feat_df.shape[1]
	features, label = pd_common_index_rows(feat_df.dropna(axis=0, how='all'), shift_label(lab_df.loc[:, lab_name]).dropna())

	# label[label==-1] = 0
	logging.info('DATA DESCRIPTION')
	logging.info('label[{name}]: \n{vc}'.format(name=lab_name, vc=label.value_counts(normalize=True, sort=True).to_frame().T))
	logging.info('num features: {}'.format(num_features))

	feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, label, train_ratio=.8)

	model = Sequential()
	model.add(LSTM(input_shape = (EMB_SIZE,), input_dim=EMB_SIZE, output_dim=HIDDEN_RNN, return_sequences=True))
	model.add(LSTM(input_shape = (EMB_SIZE,), input_dim=EMB_SIZE, output_dim=HIDDEN_RNN, return_sequences=False))
	model.add(Dense(2))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam', 
				  loss='binary_crossentropy', 
				  metrics=['accuracy'])

	logging.info('BEFORE')
	if (in_debug_mode()):
		for idx, layer in enumerate(model.layers):
			logging.debug('layer[{idx}] weights: \n{weights}'.format(idx=idx, weights=str(layer.get_weights())))

	model.fit(x=feat_train, y=lab_train, epochs=20, batch_size=128)
	score = model.evaluate(feat_test, lab_test, batch_size=128)

	logging.info('AFTER')
	print('summary: {summary}'.format(summary=str(model.summary())))
	print('{metrics}: {score}'.format(metrics=str(model.metrics_names), score=str(score)))

	if (in_debug_mode()):
		for idx, layer in enumerate(model.layers):
			logging.debug('layer[{idx}] weights: \n{weights}'.format(idx=idx, weights=str(layer.get_weights())))
		forecast = model.predict(feat_test, batch_size=128)
		logging.debug('actual: {actual} \nforecast: {forecast}'.format(actual=str(lab_test), forecast=str(forecast.T[0])))

	return score


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		net_test(sys.argv[1:])
