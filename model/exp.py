"""
Kevin Patel
"""
import sys
import os
from os.path import sep, basename, dirname, exists
import logging

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from common_util import MODEL_DIR, rectify_json, load_json, dump_json, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, dt_now, get_cmd_args
from model.common import ASSETS, EXP_DIR
from model.exp_util import get_model, get_param_dir, get_trial_dir, get_callbacks, get_trainer
from data.pl_xgdm import XGDataModule

PLSEED = dt_now().timestamp()
MAX_EPOCHS = 200


def exp(argv):
	"""
	Main experiment script.
	"""
	cmd_arg_list = ['dry-run', 'assets=', 'xdata=', 'ydata=', 'smodel=', 'models=', 'param=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__),
		script_pkg=basename(dirname(__file__)))
	dry_run = cmd_input['dry-run']
	logging.info(f'dry-run: {dry_run}')

	# data args
	if (is_valid(asset_names := cmd_input['assets='])):
		asset_names = asset_names.split(',')
	else:
		asset_names = ASSETS
	# 'price', 'ivol'
	feature_name = cmd_input['xdata='] or 'price,ivol'
	# ret: 'ret_daily_R', 'ret_daily_r'
	# rvol: 'rvol_daily_hl', 'rvol_daily_r_abs', 'rvol_daily_r²'
	#       'rvol_minutely_r_std', 'rvol_minutely_r²_sum'
	target_name = cmd_input['ydata='] or 'rvol_minutely_r²_sum'

	# model args
	sm_name = cmd_input['smodel='] or 'anp'
	if (is_valid(model_names := cmd_input['models='])):
		model_names = model_names.split(',')
	elif (sm_name == 'anp'):
		model_names = ['base', 'cnp', 'lnp', 'np']

	# param args
	logging.info('loading global training params...')
	params_t = load_json('params_t.json', EXP_DIR +sm_name +sep)
	param_name = cmd_input['param='] or '001'
	param_dir = get_param_dir(sm_name, param_name)

	# splits = ('train', 'val', 'test')
	splits = ('train', 'val')

	logging.info(f'{asset_names=}')
	logging.info(f'{feature_name=}')
	logging.info(f'{target_name=}')
	logging.info(f'model: {sm_name}->{model_names}')
	logging.info('cuda: {}'.format('✓' if (torch.cuda.is_available()) else '🞩'))

	for asset_name in asset_names:
		logging.info(f'{asset_name=}')
		logging.info('loading data...')
		dm = XGDataModule(params_t, asset_name=asset_name,
			feature_name=feature_name, target_name=target_name)
		dm.prepare_data()
		dm.setup()

		for model_name in model_names:
			logging.info(f'{model_name=}')

			if (exists(param_dir +'params_t.json')):
				logging.info('updating training params...')
				params_t = load_json('params_t.json', param_dir)
				dm.update(params_t) # update training params and re-call setup
			logging.info('loading model params...')
			params_m = load_json('params_m.json', param_dir)
			trial_dir = get_trial_dir(param_dir, model_name, dm.name, str(PLSEED))
			makedir_if_not_exists(trial_dir)

			# Build model
			model = get_model(params_m, params_t, sm_name, model_name, dm, splits)
			callbacks = get_callbacks(trial_dir, model.model_type)
			trainer = get_trainer(trial_dir, callbacks, params_t['epochs'], MAX_EPOCHS,
				model.get_precision(), PLSEED)

			if (dry_run):
				# print(f'{params_t=}\n{params_m=}')
				logging.info('dry-run: continuing loop without model fit/eval')
				continue

			trainer.fit(model, datamodule=dm)

			if ('test' in splits):
				trainer.test(model, datamodule=dm, verbose=False)

			# Dump metadata and results
			model.dump_plots(trial_dir, model_name, dm)
			model.dump_results(trial_dir, model_name)

		torch.cuda.empty_cache()

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		exp(sys.argv[1:])

