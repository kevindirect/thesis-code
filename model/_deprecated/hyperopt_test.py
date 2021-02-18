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
from model.common import DATASET_DIR, default_model, default_backend, default_dataset, default_trials_count
from model.model_util import BINARY_CLF_MAP
from model.data_util import datagen, prepare_transpose_data, prepare_label_data, prepare_target_data
from recon.dataset_util import prep_dataset
from recon.split_util import pd_binary_clip


def hyperopt_test(argv):
	cmd_arg_list = ['model=', 'backend=', 'dataset=', 'assets=', 'trials_count=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	model_code = cmd_input['model='] if (cmd_input['model='] is not None) else default_model
	backend_name = cmd_input['backend='] if (cmd_input['backend='] is not None) else default_backend
	dataset_fname = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None
	trials_count = int(cmd_input['trials_count=']) if (cmd_input['trials_count='] is not None) else default_trials_count

	mod = BINARY_CLF_MAP[backend_name][model_code]
	mod_name = mod.__name__
	dataset_name = dataset_fname[:-JSON_SFX_LEN]
	dataset_dict = load_json(dataset_fname, dir_path=DATASET_DIR)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map=None)

	logging.info('model: {}'.format(mod_name))
	logging.info('backend: {}'.format(backend_name))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']['dfs']), dataset_name))
	logging.info('assets: {}'.format(str('all' if (assets==None) else ', '.join(assets))))

	logging.info('executing...')
	for i, (fpath, lpath, _, frec, lrec, _, fcol, lcol, _, feature, label, _) in enumerate(datagen(dataset, feat_prep_fn=prepare_transpose_data, label_prep_fn=prepare_label_data, target_prep_fn=prepare_target_data, how='ser_to_ser')):
		asset_name = fpath[0]
		assert(asset_name==lpath[0])		
		logging.info('parent exp {asset} {idx}: (X, y) -> ({fcol}, {lcol})'.format(asset=asset_name, idx=i, fcol=fcol, lcol=lcol))
		pos_label, neg_label = pd_binary_clip(label)

		logging.info('pos dir model experiment')
		run_trials(mod, feature, pos_label, trials_count)
		sys.exit(0)
		logging.info('neg dir model experiment')
		run_trials(mod, feature, neg_label, trials_count)

def run_trials(model_exp, features, label, max_evals):
	exp = model_exp()
	trials = Trials()
	obj = exp.make_const_data_objective(features, label)
	best = fmin(obj, exp.get_space(), algo=tpe.suggest, max_evals=max_evals, trials=trials)
	best_params = exp.params_idx_to_name(best)

	try:
		print(trials)
	except:
		pass

	try:
		print('trials', trials.trials)
	except:
		pass

	try:
		print('results', trials.results)
	except:
		pass

	try:
		print('losses', trials.losses())
	except:
		pass

	print('best idx: {}'.format(best))
	print('best params: {}'.format(best_params))

	return best_params


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		hyperopt_test(sys.argv[1:])