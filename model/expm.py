import sys
import os
from os.path import sep, basename, dirname, exists
import logging

import numpy as np
import pandas as pd
import torch

from common_util import MODEL_DIR, rectify_json, load_json, dump_json, dump_df, benchmark, is_valid, isnt, get_cmd_args, dt_now
from model.common import ASSETS, EXP_DIR
from model.exp_util import get_param_dir, get_study_dir, run_dump_exp
from data.pl_xgdm import XGDataModule


def expm(argv):
	"""
	Manual experiment script
	"""
	cmd_arg_list = ['dry-run', 'assets=', 'xdata=', 'ydata=', 'smodel=', 'models=', 'param=', 'final']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__),
		script_pkg=basename(dirname(__file__)))
	dry_run = cmd_input['dry-run']
	logging.info(f'dry-run: {dry_run}')
	splits = ('train', 'val', 'test') if cmd_input['final'] else ('train', 'val')

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
		model_names = ['base', 'cnp', 'lnp', 'np']

	logging.info('loading training and model params...')
	param_name = cmd_input['param='] or '000'
	param_dir = get_param_dir(sm_name, param_name, dir_path=EXP_DIR +'manual' +sep)
	params_d = load_json('params_d.json', param_dir) # fixed
	params_m = load_json('params_m.json', param_dir) # fixed

	logging.info(f'{asset_names}, {feature_name=}, {target_name=}')
	logging.info(f'model: {sm_name}->{model_names}[{param_name}]')
	logging.info('cuda: {}'.format('✓' if (torch.cuda.is_available()) else '🞩'))

	for asset_name in asset_names:
		logging.info(f'{asset_name=}')
		logging.info('loading data...')
		dm = XGDataModule(params_d, asset_name=asset_name,
			feature_name=feature_name, target_name=target_name)
		dm.prepare_data()
		dm.setup()
		seed = dt_now().timestamp()

		for model_name in model_names:
			logging.info(f'{model_name=}')
			if (dry_run):
				logging.info('dry-run: skip model fit')
			else:
				study_dir = get_study_dir(param_dir, model_name, dm.name)
				run_dump_exp(study_dir, params_m, params_d, sm_name, model_name, splits, dm, seed=seed)
		torch.cuda.empty_cache()

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		expm(sys.argv[1:])

