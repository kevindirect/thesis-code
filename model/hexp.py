"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import basename
import subprocess
import logging

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials

from common_util import REPORT_DIR, JSON_SFX_LEN, makedir_if_not_exists, get_class_name, str_to_list, get_cmd_args, load_json, benchmark
from model.common import DATASET_DIR, HOPT_WORKER_BIN, TRIALS_COUNT, default_model, default_dataset
from model.model_util import BINARY_CLF_MAP
from model.data_util import datagen, prepare_transpose_data, prepare_label_data
from recon.dataset_util import prep_dataset
from recon.split_util import pd_binary_clip
from report.mongo_server import MongoServer


def hexp(argv):
	cmd_arg_list = ['model=', 'dataset=', 'assets=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	mod_code = cmd_input['model='] if (cmd_input['model='] is not None) else default_model
	dataset_fname = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None

	mod = BINARY_CLF_MAP[mod_code]()
	mod_name = get_class_name(mod)
	dataset_name = dataset_fname[:-JSON_SFX_LEN]
	dataset_dict = load_json(dataset_fname, dir_path=DATASET_DIR)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map=None)

	logging.info('model: {}'.format(mod_name))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']), dataset_name))
	logging.info('assets: {}'.format(str('all' if (assets==None) else ', '.join(assets))))

	with MongoServer() as db:
		for fpath, lpath, frec, lrec, fcol, lcol, feature, label in datagen(dataset, feat_prep_fn=prepare_transpose_data, label_prep_fn=prepare_label_data, how='ser_to_ser'):
			asset = fpath[0]
			assert(asset==lpath[0])
			group_name = '{asset},{dataset},{model}'.format(asset=asset, dataset=dataset_name, model=mod_name)
			exp_name = '{fdf}[{fcol}],{ldf}[{lcol}]'.format(fdf=frec.desc, fcol=fcol, ldf=lrec.desc, lcol=lcol)
			logging.info('{group}: {exp}'.format(group=group_name, exp=exp_name))

			pos_label, neg_label = pd_binary_clip(label)
			run_exp(mod, feature, pos_label, db, group_name, exp_key='{},pos'.format(exp_name))
			run_exp(mod, feature, neg_label, db, group_name, exp_key='{},neg'.format(exp_name))
			break

def run_exp(exp, features, label, db, db_name, exp_key='', max_evals=TRIALS_COUNT/4):
	logdir = sep.join([REPORT_DIR, *db_name.split(','), exp_key])
	makedir_if_not_exists(logdir)
	obj = exp.make_const_data_objective(features, label, logdir)
	logging.info('{group}: {exp}'.format(group=db_name, exp=exp_key))

	if (db is not None):
		trials = MongoTrials(db.get_mongodb_trials_uri(db_name=db_name), exp_key=exp_key)
		worker_args = [HOPT_WORKER_BIN]
		worker_args.append('--max-jobs={max_jobs}'.format(max_jobs=1))
		worker_args.append('--mongo={db_uri}'.format(db_uri=db.get_mongodb_uri(db_name=db_name)))
		worker_args.append('--poll-interval={poll_interval}'.format(poll_interval=0.1))
		worker_args.append('--workdir={dir}'.format(dir=sep.join([REPORT_DIR, 'workdir'])))
		worker = subprocess.Popen(worker_args, stdout=db.fnull, stderr=subprocess.STDOUT, env=os.environ.copy(), shell=False)
	else:
		trials = Trials()
	best = fmin(obj, exp.get_space(), algo=tpe.suggest, max_evals=max_evals, trials=trials)
	print('best idx: {}'.format(best))
	# best_params = exp.params_idx_to_name(best)
	# print('best params: {}'.format(best_params))


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		hexp(sys.argv[1:])
