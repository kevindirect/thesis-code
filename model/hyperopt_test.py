"""
Kevin Patel
"""
import sys
import os
from os.path import basename
import logging

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, Trials

from common_util import JSON_SFX_LEN, get_class_name, str_to_list, get_cmd_args, pd_common_index_rows, load_json, benchmark
from model.common import DATASET_DIR, default_model, default_dataset
from model.model_util import BINARY_CLF_MAP
from model.data_util import datagen, prepare_transpose_data, prepare_label_data
from recon.dataset_util import prep_dataset
from recon.split_util import pd_binary_clip


def hyperopt_test(argv):
	cmd_arg_list = ['model=', 'dataset=', 'assets=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	mod_code = cmd_input['model='] if (cmd_input['model='] is not None) else default_model
	dataset_fname = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None

	mod = BINARY_CLF_MAP[mod_code]
	mod_name = mod.__name__
	dataset_name = dataset_fname[:-JSON_SFX_LEN]
	dataset_dict = load_json(dataset_fname, dir_path=DATASET_DIR)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map=None)

	logging.info('model: {}'.format(mod_name))
	logging.info('assets: {}'.format(str('all' if (assets==None) else ', '.join(assets))))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']), dataset_name))

	logging.info('executing...')
	for fpath, lpath, frec, lrec, fcol, lcol, feature, label in datagen(dataset, feat_prep_fn=prepare_transpose_data, label_prep_fn=prepare_label_data, how='ser_to_ser'):
		logging.info('(X, y) -> ({fcol}, {lcol})'.format(fcol=fcol, lcol=lcol))
		pos_label, neg_label = pd_binary_clip(label)

		logging.info('pos dir model experiment')
		run_trials(mod, feature, pos_label)
		sys.exit(0)
		logging.info('neg dir model experiment')
		run_trials(mod, feature, neg_label)

def run_trials(model_exp, features, label):
	exp = model_exp()
	trials = Trials()
	obj = exp.make_const_data_objective(features, label, '')
	best = fmin(obj, exp.get_space(), algo=tpe.suggest, max_evals=5, trials=trials)
	best_params = exp.params_idx_to_name(best)
	print(trials)
	print('trials', trials.trials)
	print('results', trials.results)
	print('losses', trials.losses())

	print('best idx: {}'.format(best))
	print('best params: {}'.format(best_params))

	return best_params


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		hyperopt_test(sys.argv[1:])
