"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import basename
import copy
import time
import subprocess
import logging

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials

from common_util import CRUNCH_DIR, REPORT_DIR, ALL_COLS, makedir_if_not_exists, get_class_name, str_to_list, get_cmd_args, load_json, benchmark
from model.common import XG_DIR, HOPT_WORKER_BIN, default_model, default_backend, default_xg, default_trials_count
from model.model_util import CLF_MAP
from model.data_util import xgdg
from recon.dataset_util import prep_dataset
from recon.split_util import pd_binary_clip
from report.mongo_server import MongoServer


def xgexp(argv):
	cmd_arg_list = ['model=', 'backend=', 'xg=', 'assets=', 'trials_count=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	model_code = cmd_input['model='] if (cmd_input['model='] is not None) else default_model
	backend_name = cmd_input['backend='] if (cmd_input['backend='] is not None) else default_backend
	xg_fname = cmd_input['xg='] if (cmd_input['xg='] is not None) else default_xg
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None
	trials_count = int(cmd_input['trials_count=']) if (cmd_input['trials_count='] is not None) else default_trials_count

	mod_obj = CLF_MAP[backend_name][model_code]()
	mod_name = get_class_name(mod_obj)

	logging.info('model: {}'.format(mod_name))
	logging.info('backend: {}'.format(backend_name))
	logging.info('xg: {}'.format(xg_fname))
	logging.info('assets: {}'.format(str('all' if (assets==None) else ', '.join(assets))))

	with MongoServer() as db:
		for i, (paths, recs, dfs) in enumerate(xgdg(xg_fname, delayed=True, assets=assets, filters_map=None)):
			logging.info('parent exp {}'.format(i))
			asset_name = paths[0][0]
			assert(asset_name==paths[1][0]==paths[2][0])
			frec, lrec, trec = recs
			meta = {
				'group': {
					'name': '{asset},{xg},{model}_{backend0}'.format(asset=asset_name, xg=xg_fname, model=mod_name, backend0=backend_name[0]),
					'asset': asset_name,
					'xg': xg_fname.rstrip('.json'),
					'backend': backend_name,
					'model': mod_name
				},
				'exp': {
					'name': '{feat},{lab},{dir}',
					'feat': '{featdf}[{featcol}]'.format(featdf=frec.desc, featcol=ALL_COLS),
					'lab': '{labdf}[{labcol}]'.format(labdf=lrec.desc, labcol=ALL_COLS),
					'dir': '{dir}'
				}
			}
			f, l, t = dfs.compute()
			pos_l, neg_l = pd_binary_clip(l)
			pos_meta, neg_meta = copy.deepcopy(meta), copy.deepcopy(meta)
			pos_meta['exp']['dir'], neg_meta['exp']['dir'] = 'pos', 'neg'
			pos_meta['exp']['name'], neg_meta['exp']['name'] = pos_meta['exp']['name'].format(**pos_meta['exp']), neg_meta['exp']['name'].format(**neg_meta['exp'])
			run_model(mod_obj, f, pos_l, pos_meta, db, trials_count)
			run_model(mod_obj, f, neg_l, neg_meta, db, trials_count)


def run_model(mdl, features, labels, meta, db, max_evals):
	"""
	Run the model over passed (features, labels) using metadata in meta.
	"""
	db_name, exp_name = meta['group']['name'], meta['exp']['name']
	logdir = REPORT_DIR +sep.join([*db_name.split(','), *exp_name.split(',')]) +sep
	makedir_if_not_exists(logdir)
	obj = mdl.make_const_data_objective(features, labels, logdir, exp_meta=meta)
	logging.info('{group}: {exp}'.format(group=db_name, exp=exp_name))

	if (db is not None):
		worker_args = [HOPT_WORKER_BIN]
		worker_args.append('--exp-key={exp}'.format(exp=exp_name))					# XXX - only run this experiment
		worker_args.append('--max-jobs={max_jobs}'.format(max_jobs=max_evals))		# XXX - set jobs to number of trials
		worker_args.append('--mongo={db_uri}'.format(db_uri=db.get_mongodb_uri(db_name=db_name)))
		worker_args.append('--poll-interval={poll_interval:1.2f}'.format(poll_interval=0.1))
		worker_args.append('--workdir={dir}'.format(dir=CRUNCH_DIR))
		worker = subprocess.Popen(worker_args, stdout=db.fnull, stderr=subprocess.STDOUT, shell=False)
		logging.info('started worker: {}'.format(' '.join(worker_args)))
		trials = MongoTrials(db.get_mongodb_trials_uri(db_name=db_name), exp_key=exp_name)
		time.sleep(3)
	else:
		trials = Trials()
	best = fmin(obj, mdl.get_space(), algo=tpe.suggest, max_evals=max_evals, trials=trials)
	print('best idx: {}'.format(best))
	# best_params = exp.params_idx_to_name(best)
	# print('best params: {}'.format(best_params))


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		xgexp(sys.argv[1:])
