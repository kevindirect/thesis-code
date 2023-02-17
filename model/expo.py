import sys
import os
from os.path import sep, basename, dirname, exists
import logging

import numpy as np
import pandas as pd
import torch
import optuna

from common_util import MODEL_DIR, rectify_json, load_json, dump_df, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args
from model.common import ASSETS, EXP_DIR, OPTUNA_DBNAME, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT
from model.exp_util import get_optmode, get_param_dir, get_study_dir, get_study_name, get_objective_fn
from data.pl_xgdm import XGDataModule
from model.optuna_util import get_sampler, get_model_suggestor


def expo(argv):
	"""
	Optuna experiment script
	"""
	cmd_arg_list = ['dry-run', 'assets=', 'xdata=', 'ydata=', 'smodel=', 'models=', 'param=', 'obj=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__),
		script_pkg=basename(dirname(__file__)))
	dry_run = cmd_input['dry-run']
	logging.info(f'dry-run: {dry_run}')
	splits = ('train', 'val')

	# data args
	if (is_valid(asset_names := cmd_input['assets='])):
		asset_names = asset_names.split(',')
	else:
		asset_names = ASSETS
	# 'price', 'ivol', 'logprice', 'logivol', 'logchangeprice', 'logchangeivol'
	feature_name = cmd_input['xdata='] or 'logchangeprice,logchangeivol'
	# ret: 'R_1day', 'r_1day'
	# rvol: 'rvol_1day_r_1min_{meanad, std, var, abs, rms}', 'rvol_1day_r²_1min_sum'
	target_name = cmd_input['ydata='] or 'rvol_1day_r_1min_std'

	# model args
	sm_name = cmd_input['smodel='] or 'anp'
	if (is_valid(model_names := cmd_input['models='])):
		model_names = model_names.split(',')
	elif (sm_name == 'anp'):
		model_names = ['np']

	# optuna args
	obj = cmd_input['obj='] or 'val_reg_mae' # val_loss, val_kldiv, val_reg_mae, val_reg_mse
	optmode = get_optmode(obj)
	sampler_type = 'tpe'

	logging.info('loading training and model params...')
	param_name = cmd_input['param='] or '000'
	param_dir = get_param_dir(sm_name, param_name, dir_path=EXP_DIR+sampler_type+sep)
	params_d = load_json('params_d.json', param_dir) # fixed
	params_m = load_json('params_m.json', param_dir) # overwritten by optuna suggestions

	logging.info(f'{asset_names}, {feature_name=}, {target_name=}')
	logging.info(f'model: {sm_name}->{model_names}[{param_name}]')
	logging.info('cuda: {}'.format('✓' if (torch.cuda.is_available()) else '🞩'))
	logging.info(f'{optmode}: {obj}')

	for asset_name in asset_names:
		logging.info(f'{asset_name=}')
		logging.info('loading data...')
		dm = XGDataModule(params_d, asset_name=asset_name,
			feature_name=feature_name, target_name=target_name)
		dm.prepare_data()
		dm.setup()

		for model_name in model_names:
			logging.info(f'{model_name=}')
			study_dir = get_study_dir(param_dir, model_name, dm.name) +obj +sep
			study_name = get_study_name(study_dir, dir_path=EXP_DIR)
			study_db = f'sqlite:///{study_dir}{OPTUNA_DBNAME}.db'
			makedir_if_not_exists(study_dir)

			sampler = get_sampler(sm_name, model_name, sampler_type)
			study = optuna.create_study(storage=study_db, load_if_exists=True,
				sampler=sampler, direction=optmode, study_name=study_name)
			suggestor_m = get_model_suggestor(sm_name, model_name)
			objective_fn = get_objective_fn(study_dir, params_m, params_d, sm_name,
				model_name, splits, dm, obj, suggestor_m)
			if (dry_run):
				logging.info('dry-run: skip study optimize')
			else:
				study.optimize(objective_fn, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT*60,
					catch=(), n_jobs=1, gc_after_trial=False, show_progress_bar=False)
				study_df = study.trials_dataframe().sort_values(by='value', ascending=True)
				dump_df(study_df.set_index('number'), f'{OPTUNA_DBNAME}.csv', dir_path=study_dir, data_format='csv')
		torch.cuda.empty_cache()

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		expo(sys.argv[1:])

