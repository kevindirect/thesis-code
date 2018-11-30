"""
Kevin Patel
"""

import sys
import os
from os import sep
from os.path import splitext
from itertools import product
from functools import partial, reduce
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute, visualize
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from common_util import MODEL_DIR, RECON_DIR, JSON_SFX_LEN, str_to_list, get_cmd_args, pd_common_index_rows, load_json, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, default_dataset, default_opt_filter
from model.model_util import datagen, prepare_transpose_data, prepare_label_data
from model.model.ThreeLayerBinaryFFN import ThreeLayerBinaryFFN
from model.model.OneLayerBinaryLSTM import OneLayerBinaryLSTM
from recon.dataset_util import prep_dataset
from recon.split_util import get_train_test_split, pd_binary_clip


def hyperopt_test(argv):
	cmd_arg_list = ['dataset=', 'filterset=', 'idxfilters=', 'assets=', 'target_idx=', 'visualize']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='hyperopt_test')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	filterset_name = cmd_input['filterset='] if (cmd_input['filterset='] is not None) else '_'.join(['default', dataset_name])
	filter_idxs = str_to_list(cmd_input['idxfilters=']) if (cmd_input['idxfilters='] is not None) else default_opt_filter
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None
	run_compute = True if (cmd_input['visualize'] is None) else False

	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	filter_dict = load_json(filterset_name, dir_path=FILTERSET_DIR)

	filterset = []
	for filter_idx in filter_idxs:
		selected = [flt for flt in filter_dict[filter_idx] if (flt not in filterset)]
		filterset.extend(selected)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map={'features': filterset})

	logging.info('assets: {}'.format(str('all' if (assets==None) else ', '.join(assets))))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']), dataset_name[:-JSON_SFX_LEN]))
	logging.info('filter: {} [{}]'.format(filterset_name[:-JSON_SFX_LEN], str(', '.join(filter_idxs))))
	logging.debug('filterset: {}'.format(filterset))

	if (run_compute):
		logging.info('executing...')
		for feature, label in datagen(dataset, feat_prep_fn=prepare_transpose_data, label_prep_fn=prepare_label_data, how='ser_to_ser'):
			pos_label, neg_label = pd_binary_clip(label)
			f, lpos, lneg = pd_common_index_rows(feature, pos_label, neg_label)

			logging.info('pos dir model experiment')
			run_trials(ThreeLayerBinaryFFN, f, lpos)

			logging.info('neg dir model experiment')
			run_trials(ThreeLayerBinaryFFN, f, lneg)

def run_trials(model_exp, features, label):
	exp = model_exp()
	trials = Trials()
	obj = exp.make_const_data_objective(features, label)
	best = fmin(obj, exp.get_space(), algo=tpe.suggest, max_evals=50, trials=trials)
	best_params = exp.params_idx_to_name(best)
	bad = exp.get_bad_trials()

	print('best idx: {}'.format(best))
	print('best params: {}'.format(best_params))
	if (bad > 0):
		print('bad trials: {}'.format(bad))

	return best_params


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		hyperopt_test(sys.argv[1:])
