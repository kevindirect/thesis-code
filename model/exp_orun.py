"""
Kevin Patel
"""
import sys
import os
from os.path import sep, basename, dirname, exists
from functools import partial
import logging

import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
from verification.batch_norm import BatchNormVerificationCallback
from verification.batch_gradient import BatchGradientVerificationCallback
import optuna
from optuna.pruners import PercentilePruner, HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback

from common_util import MODEL_DIR, load_df, deep_update, rectify_json, load_json, dump_json, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args
from model.common import ASSETS, INTERVAL_YEARS, WIN_SIZE, OPTUNA_DB_FNAME, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_HOURS, INTRADAY_LEN, EXP_LOG_DIR, EXP_PARAMS_DIR
from model.pl_xgdm import XGDataModule
from model.exp_run import dump_benchmarks, get_train_params, get_model_params, get_optmode, get_study_dir, get_trial_dir, get_model, get_trainer
from model.opt_util import get_suggest_train, get_suggest_model


def exp_optuna_run(argv):
	cmd_arg_list = ['dry-run', 'intstart=', 'file=', 'assets=', 'smodel=', 'models=', 'xdata=', 'ydata=', 'objectives=', 'guide=', 'trials=', 'run-hours=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__),
		script_pkg=basename(dirname(__file__)))
	dry_run = cmd_input['dry-run']
	logging.info(f'dry-run: {dry_run}')

	# data args
	int_years = (int(cmd_input['intstart='] or INTERVAL_YEARS[0]), INTERVAL_YEARS[1])
	fdata_name = cmd_input['xdata='] or 'h_pba_h'
	ldata_name = cmd_input['ydata='] or 'ddir'
	if (is_valid(asset_names := cmd_input['assets='])):
		asset_names = asset_names.split(',')
	else:
		asset_names = ASSETS
	splits = ('train', 'val')
	# splits = ('train', 'val', 'test')

	# model args
	params_dict = load_json(cmd_input['file='] or 'exp_orun.json', MODEL_DIR)
	sm_name = cmd_input['smodel='] or 'anp'
	if (is_valid(model_names := cmd_input['models='])):
		model_names = model_names.split(',')
	elif (sm_name == 'anp'):
		model_names = ['np']

	# optuna args
	if (is_valid(objectives := cmd_input['objectives='])):
		objectives = objectives.split(',')
	else:
		# objectives = ['val_binary_long_sharpe', 'val_binary_short_sharpe'] # error: reporting doesnt work with multiobj
		objectives = ['val_binary_sharpe']
	optmodes = [get_optmode(obj) for obj in objectives]
	monitor = cmd_input['guide='] or 'val_binary_sharpe' # pruning guide / monitor
	n_trials = int(cmd_input['trials=']) if (cmd_input['trials=']) else OPTUNA_N_TRIALS
	hourly_timeout = int(cmd_input['run-hours=']) if (cmd_input['run-hours=']) \
		else OPTUNA_TIMEOUT_HOURS

	exp_dir = EXP_LOG_DIR +sep.join(['optuna', sm_name]) +sep
	logging.info(f'{int_years=}')
	logging.info(f'{fdata_name=}')
	logging.info(f'{ldata_name=}')
	logging.info(f'{asset_names=}')
	logging.info(f'model: {sm_name}->{model_names}')
	logging.info(f'{n_trials=} for {hourly_timeout} hour(s)')
	logging.info(f'{objectives=}, {optmodes=}')
	logging.info(f'{monitor=}')
	logging.debug('cuda: {}'.format('✓' if (torch.cuda.is_available()) else '🞩'))

	# loss_type = t_params['loss']
	# model_type = loss_type.split('-')[0]
	# num_classes = 2 if (model_type == 'clf') else None
	# logging.getLogger("lightning").setLevel(logging.ERROR) # Disable pl warnings

	for asset_name in asset_names:
		logging.info(f'{asset_name=}')
		asset_dir = f'{exp_dir}{asset_name}{sep}'

		logging.info('loading training params...')
		dm_name = XGDataModule.get_name(int_years, fdata_name, ldata_name)

		for params_name in map(str, params_dict[asset_name][dm_name]):
			logging.info(f'{params_name=}')
			t_params = get_train_params(sm_name, params_name)
			t_params['epochs'] = 40 # hardcoded

			logging.info('loading data...')
			dm = XGDataModule(t_params, asset_name, fdata_name, ldata_name,
				interval=int_years, fret=None, overwrite_cache=False)
			dm.prepare_data()
			dm.setup()
			dump_benchmarks(asset_name, dm)

			for model_name in model_names:
				logging.info(f'{model_name=}')

				logging.info('loading model params...')
				m_params = get_model_params(sm_name, params_name, asset_name, model_name)

				if (dry_run):
					# print(f'{t_params=}')
					# print(f'{m_params=}')
					logging.info('dry-run: continuing study loop')
					continue

				# optuna study setup
				study_dir = get_study_dir(f'{asset_dir}{model_name}{sep}',
					dm, params_name) +','.join(objectives) +sep
				study_name = get_ostudy_name(study_dir)
				makedir_if_not_exists(study_dir)
				study_db = f'sqlite:///{study_dir}{OPTUNA_DB_FNAME}'
				logging.info(f'{study_name=}')
				logging.debug(f'{study_dir=}')
				logging.debug(f'{study_db=}')
				logging.debug(f'gpu mem (mb): {torch.cuda.max_memory_allocated()}')

				sampler = optuna.samplers.TPESampler(multivariate=True)
				pruner = HyperbandPruner(min_resource=1, max_resource='auto',
					reduction_factor=3, bootstrap_count=0)
				study = optuna.create_study(storage=study_db, load_if_exists=True,
					sampler=sampler, pruner=pruner, directions=optmodes,
					study_name=study_name)

				if (len(study.trials) == 0):
					# Freshly created study, optimize base params:
					study.enqueue_trial({**t_params, **m_params})
					run_obj = partial(run_fixed_objective, objectives=objectives, monitor=monitor,
						study_dir=study_dir, sm_name=sm_name, model_name=model_name, asset_name=asset_name,
						dm=dm, m_params=m_params, t_params=t_params, splits=splits)
					study.optimize(run_obj, n_trials=1)

				get_obj = partial(run_suggest_objective, objectives=objectives, monitor=monitor,
					study_dir=study_dir, sm_name=sm_name, model_name=model_name, asset_name=asset_name,
					dm=dm, m_params=m_params, t_params=t_params, splits=splits)
				study.optimize(get_obj, n_trials=n_trials, timeout=hourly_timeout*60*60,
					catch=(), n_jobs=1, gc_after_trial=False, show_progress_bar=False)

		torch.cuda.empty_cache()

def get_ostudy_name(study_dir, base_dir=EXP_LOG_DIR+'optuna'+sep):
	return study_dir.replace(base_dir, '').replace(sep, ',')

def get_ocallbacks(trial, trial_dir, monitor):
	mode = get_optmode(monitor)[:3]
	chk_callback = pl.callbacks.ModelCheckpoint(f'{trial_dir}chk{sep}',
		monitor=monitor, mode=mode)
	es_callback = PyTorchLightningPruningCallback(trial, monitor=monitor)
	# ver_callbacks = (BatchNormVerificationCallback(),
		# BatchGradientVerificationCallback())
	callbacks = [chk_callback, es_callback]
	return callbacks

def run_suggest_objective(trial, objectives, monitor, study_dir,
	sm_name, model_name, asset_name, dm, m_params, t_params, splits):
	suggest_update(trial, sm_name, model_name, asset_name, dm, m_params, t_params)
	return run_fixed_objective(trial, objectives, monitor, study_dir,
		sm_name, model_name, asset_name, dm, m_params, t_params, splits)

def run_fixed_objective(trial, objectives, monitor, study_dir,
	sm_name, model_name, asset_name, dm, m_params, t_params, splits):
	trial_dir = f'{study_dir}{str(trial.number).zfill(6)}{sep}'
	makedir_if_not_exists(trial_dir)
	logging.debug(f'{trial_dir=}')
	dump_json(rectify_json(m_params), 'params_m.json', trial_dir)
	dump_json(rectify_json(t_params), 'params_t.json', trial_dir)

	model = get_model(sm_name, m_params, t_params, dm, splits)
	callbacks = get_ocallbacks(trial, trial_dir, monitor=monitor)
	trainer = get_trainer(trial_dir, callbacks, t_params['epochs'],
		model.get_precision())
	trainer.fit(model, datamodule=dm)

	# Dump metadata and results
	model.dump_plots(trial_dir, model_name, dm)
	results = model.dump_results(trial_dir, model_name)

	return get_scores(results, objectives)

def suggest_update(trial, sm_name, model_name, asset_name, dm, m_params, t_params):
	m_suggest = get_suggest_model(m_params, sm_name, model_name, asset_name, dm.name)(trial)
	deep_update(m_params, m_suggest)
	# t_suggest = get_suggest_train(t_params, sm_name, model_name, asset_name, dm.name)(trial)
	# deep_update(t_params, t_suggest)
	# dm.update_params(t_params)
	logging.debug(f'{m_params=}')
	logging.debug(f'{t_params=}')

def get_scores(results, objectives):
	return tuple(results[obj.split('_')[0]][obj] for obj in objectives)

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		exp_optuna_run(sys.argv[1:])

