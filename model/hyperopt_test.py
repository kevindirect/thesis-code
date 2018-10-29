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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Merge, Input, concatenate, Dense, Activation, Dropout, LSTM, Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras import regularizers
from keras import losses

from common_util import RECON_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, str_to_list, get_cmd_args, in_debug_mode, pd_common_index_rows, load_json, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, default_dataset, default_opt_filter, default_target_idx
from model.model_util import prepare_transpose_data, prepare_masked_labels
from recon.dataset_util import prep_dataset, prep_labels, gen_group
from recon.split_util import get_train_test_split, gen_time_series_split
from recon.label_util import shift_label


def hyperopt_test(argv):
	cmd_arg_list = ['dataset=', 'filterset=', 'idxfilters=', 'assets=', 'target_idx=', 'visualize']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='hyperopt_test')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	filterset_name = cmd_input['filterset='] if (cmd_input['filterset='] is not None) else '_'.join(['default', dataset_name])
	filter_idxs = str_to_list(cmd_input['idxfilters=']) if (cmd_input['idxfilters='] is not None) else default_opt_filter
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None
	target_idx = str_to_list(cmd_input['target_idx='], cast_to=int) if (cmd_input['target_idx='] is not None) else default_target_idx
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

	dataset_grid = {
		'feat_idx': [0, 1, 2, 3, 4],
		'label_idx': [0, 1, 2]
	}

	dataset_space = {
		'feat_idx': hp.choice('feat_idx', [0, 1, 2, 3, 4]),
		'label_idx': hp.choice('label_idx', [0, 1, 2])
	}

	one_layer_lstm_space = {
		'layer1_size': hp.choice('layer1_size', [8, 16, 32, 64, 128]),
		'lr': hp.choice('lr',[0.01, 0.001, 0.0001]),
		'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear']),
		'output_activation' : hp.choice('output_activation', ['softmax', 'sigmoid', 'tanh']),
		'loss': hp.choice('loss', [losses.categorical_crossentropy])
	}

	if (run_compute):
		logging.info('executing...')
		for paths, dfs in gen_group(dataset):
			results = {}
			fpaths, lpaths, rpaths = paths
			features, labels, row_masks = dfs
			asset = fpaths[0]
			logging.info('fpaths: ' +str(fpaths))
			logging.info('lpaths: ' +str(lpaths))
			logging.info('rpaths: ' +str(rpaths))

			masked_labels = prepare_masked_labels(labels, ['bool'], labs_filter)

			for feat_idx, label_idx in product(*dataset_grid.values()):
				final_feature = prepare_transpose_data(features.iloc[:, [feat_idx]], row_masks).dropna(axis=0, how='all')
				final_label = delayed(shift_label)(final_labels.iloc[:, target_idx]).dropna()
				final_common = delayed(pd_common_index_rows)(final_feature, final_label)
				f, l = final_common.compute()

				trials = Trials()
				best = fmin(partial(one_layer_lstm, f, l), one_layer_lstm_space, algo=tpe.suggest, max_evals=50, trials=trials)
				print('best: {}'.format(best.compute()))


def one_layer_lstm(features, labels, params):

	feat_train, feat_test, lab_train, lab_test = get_train_test_split(features.values, to_categorical(label.values), train_ratio=.8, to_np=False)

	try:
		main_input = Input(shape=(feat_train.shape[0],), name='main_input')
		x = LSTM(params['layer1_size'], activation=params['activation'])(main_input)
		output = Dense(1, activation = params['output_activation'], name='output')(x)
		final_model = Model(inputs=[main_input], outputs=[output])
		opt = Adam(lr=params['lr'])

		final_model.compile(optimizer=opt,  loss=params['loss'])

		model_results = final_model.fit(feat_train, lab_train, 
					epochs=5, 
					batch_size=256, 
					verbose=1, 
					validation_data=(feat_test, lab_test),
					shuffle=False)

		return {'loss': history.history['val_loss'], 'status': STATUS_OK}

	except:
		logging.error('Error ocurred during experiment')
		return {'loss': 999999, 'status': STATUS_OK}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		hyperopt_test(sys.argv[1:])
