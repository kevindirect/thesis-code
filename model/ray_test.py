"""
Kevin Patel
"""
import sys
import os
from os import sep
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK
import ray
from ray.tune import run_experiments, register_trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch

from common_util import MODEL_DIR, REPORT_DIR, JSON_SFX_LEN, NestedDefaultDict, str_now, dump_df, makedir_if_not_exists, str_to_list, get_cmd_args, pd_common_index_rows, load_json, benchmark
from model.common import DATASET_DIR, default_model, default_dataset
from model.model_util import BINARY_CLF_MAP, datagen, prepare_transpose_data, prepare_label_data
from recon.dataset_util import prep_dataset
from recon.split_util import pd_binary_clip


def ray_test(argv):
	cmd_arg_list = ['model=', 'dataset=', 'assets=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='ray_test')
	model_name = cmd_input['model='] if (cmd_input['model='] is not None) else default_model
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None

	mod = BINARY_CLF_MAP[model_name]()
	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map=None)
	rayconfig = load_json(rayconfig_name, dir_path=MODEL_DIR)
	ray.init(**rayconfig['init'])
	index = NestedDefaultDict()

	logging.info('model: {}'.format(model_name))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']), dataset_name[:-JSON_SFX_LEN]))
	logging.info('assets: {}'.format(str('all' if (assets==None) else ', '.join(assets))))
	logging.info('starting experiment loop')

	for i, (fpath, lpath, frec, lrec, fcol, lcol, feature, label) in enumerate(datagen(dataset, feat_prep_fn=prepare_transpose_data, label_prep_fn=prepare_label_data, how='ser_to_ser')):
		assert(fpath[0]==lpath[0])
		logging.info('experiment {}'.format(i))
		asset = fpath[0]
		mod_keys = [REPORT_DIR, asset, dataset_name, 'ray', model_name]
		pos_dir = sep.join(mod_keys + ['pos'])
		neg_dir = sep.join(mod_keys + ['neg'])
		makedir_if_not_exists(pos_dir)
		makedir_if_not_exists(neg_dir)
		pos_label, neg_label = pd_binary_clip(label)
		f, lpos, lneg = pd_common_index_rows(feature, pos_label, neg_label)
		config = {
			'pos': {
				"run": mod.make_ray_objective(mod.make_const_data_objective(f, lpos)),
				"stop": {
					"timesteps_total": 100
				},
				'trial_resources'={
					"cpu": 1,
					"gpu": 1
				},
				"local_dir": sep.join([pos_dir, i])
			},
			'neg': {
				"run": mod.make_ray_objective(mod.make_const_data_objective(f, lneg)),
				"stop": {
					"timesteps_total": 100
				},
				'trial_resources'={
					"cpu": 1,
					"gpu": 1
				},
				"local_dir": sep.join([neg_dir, i])
			}
		}
		row = {
			'index': i,
			'fname': frec.name,
			'lname': lrec.name,
			'fdesc': frec.desc,
			'ldesc': lrec.desc,
			'fcol': fcol,
			'lcol': lcol,
			'start': None,
			'end': None,
			'num': 0
		}
		algo = HyperOptSearch(mod.get_space(), max_concurrent=4, reward_attr="loss")
		scheduler = AsyncHyperBandScheduler(reward_attr="loss")
		row['start'] = str_now() 
		trials = run_experiments(config, search_alg=algo, scheduler=scheduler, verbose=True)
		row['end'] = str_now()
		row['num'] = len(trials)
		try:
			index[mod_keys]
		except ValueError as e:
			index[mod_keys] = []
		finally:
			index[mod_keys].append(row)

	logging.info('dumping experiment index files...')
	for keys, val in index.items():
		index_dir = sep.join(keys)
		logging.info(index_dir)
		index_df = pd.DataFrame(val).set_index('index')
		dump_df(index_df, 'index.csv', dir_path=index_dir, data_format='csv')

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		ray_test(sys.argv[1:])
