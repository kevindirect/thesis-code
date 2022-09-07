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
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
# from verification.batch_norm import BatchNormVerificationCallback
# from verification.batch_gradient import BatchGradientVerificationCallback

from common_util import MODEL_DIR, load_df, load_json, rectify_json, dump_json, str_now, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args
from model.common import EXP_LOG_DIR, EXP_PARAMS_DIR, ASSETS, INTERVAL_YEARS, WIN_SIZE, INTRADAY_LEN
from model.pl_xgdm import XGDataModule


def exp_run(argv):
	cmd_arg_list = ['dry-run', 'intstart=', 'assets=', 'smodel=', 'models=', 'params=', 'xdata=', 'ydata=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__), \
		script_pkg=basename(dirname(__file__)))
	dry_run = cmd_input['dry-run']
	int_years = (int(cmd_input['intstart='] or INTERVAL_YEARS[0]), INTERVAL_YEARS[1])
	sm_name = cmd_input['smodel='] or 'anp'
	if (is_valid(models := cmd_input['models='])):
		model_names = ci_assets.split(',')
	elif (sm_name == 'anp'):
		model_names = ['base', 'cnp', 'lnp', 'np']
	if (is_valid(ci_assets := cmd_input['assets='])):
		asset_names = ci_assets.split(',')
	else:
		asset_names = ASSETS
	fdata_name = cmd_input['xdata='] or 'h_pba_h'
	ldata_name = cmd_input['ydata='] or 'ddir'
	logging.info(f'{int_years=}')
	logging.info(f'{fdata_name=}')
	logging.info(f'{ldata_name=}')

	files = os.listdir(EXP_PARAMS_DIR +sm_name)
	longest_name = max(map(len, files))
	params_last = sorted(filter(lambda f: len(f) == longest_name, files))[-1]
	params_name = cmd_input['params='] or params_last # hyperparam set to use
	logging.info(f'{params_name=}')

	t_params = get_train_params(sm_name, params_name)
	exp_dir = f'{EXP_LOG_DIR}{sm_name}{sep}'
	splits = ('train', 'val')
	# splits = ('train', 'val', 'test')

	for asset_name in asset_names:
		logging.info(f"{asset_name=}")
		asset_dir = f'{exp_dir}{asset_name}{sep}'

		logging.info("loading data...")
		dm = XGDataModule(t_params, asset_name, fdata_name, ldata_name,
			interval=int_years, fret=None, overwrite_cache=False)
		dm.prepare_data()
		dm.setup()
		dump_benchmarks(asset_name, dm)

		for model_name in model_names:
			logging.info(f"{model_name=}")

			if (dry_run):
				logging.info("dry-run: continuing study loop")
				continue

			# get/dump params
			m_params = get_model_params(sm_name, params_name, asset_name, model_name)
			study_dir = get_study_dir(f'{asset_dir}{model_name}{sep}', dm, params_name)
			makedir_if_not_exists(study_dir)
			dump_json(rectify_json(m_params), 'params_m.json', study_dir)
			dump_json(rectify_json(t_params), 'params_t.json', study_dir)

			# Build and train model
			makedir_if_not_exists(trial_dir := get_trial_dir(study_dir))
			model = get_model(sm_name, m_params, t_params, dm, splits)
			callbacks = get_callbacks(trial_dir)
			trainer = get_trainer(trial_dir, callbacks, t_params['epochs'],
				model.get_precision())
			trainer.fit(model, datamodule=dm)

			if ('test' in splits):
				trainer.test(model, datamodule=dm)

			# Dump metadata and results
			model.dump_plots(trial_dir, model_name, dm)
			model.dump_results(trial_dir, model_name)

		torch.cuda.empty_cache()


def dump_benchmarks(asset_name, dm):
	bench_data_name = f'{dm.interval[0]}_{dm.interval[1]}_{dm.ldata_name}'
	bench_dir = EXP_LOG_DIR +sep.join(['bench', asset_name, bench_data_name]) +sep
	if (not exists(bench_dir)):
		logging.info("dumping benchmarks...")
		makedir_if_not_exists(bench_dir)
		bench = dm.get_benchmarks()
		dm.dump_benchmarks_plots(bench, bench_dir)
		dm.dump_benchmarks_results(bench, bench_dir)
	else:
		logging.info("skipping benchmarks...")

def get_train_params(sm_name, params_name):
	params_dir = EXP_PARAMS_DIR +sep.join([sm_name, params_name]) +sep
	return load_json('params_t.json', params_dir)

def get_model_params(sm_name, params_name, asset_name, model_name):
	params_dir = EXP_PARAMS_DIR +sep.join([sm_name, params_name]) +sep
	params_asset_dir = f'{params_dir}{asset_name}{sep}'
	params_model_dir = f'{params_asset_dir}{model_name}{sep}'

	m_params_common = load_json('params_m.json', params_dir)
	m_params_asset = load_json('params_m.json', params_asset_dir)
	m_params_model = load_json('params_m.json', params_model_dir)
	return {**m_params_common, **m_params_asset, **m_params_model}

def get_study_dir(model_dir, dm, params_name):
	"""
	A study is defined as a particular combination of model, data, and hyper-parameters
	"""
	return model_dir +sep.join([dm.name, params_name]) +sep

def get_trial_dir(study_dir):
	"""
	A trial is a run of a study
	"""
	trial_time = str_now().replace(' ', '_').replace(':', '-')
	return f'{study_dir}{trial_time}{sep}'

def get_model(sm_name, m_params, t_params, dm, splits):
	if (sm_name in ('stcn', 'StackedTCN', 'GenericModel_StackedTCN')):
		from model.pl_generic import GenericModel
		from model.model_util import StackedTCN
		pl_model_fn, pt_model_fn = GenericModel, StackedTCN
	elif (sm_name in ('anp', 'AttentiveNP', 'NPModel_AttentiveNP')):
		from model.pl_np import NPModel
		from model.np_util2 import AttentiveNP
		pl_model_fn, pt_model_fn = NPModel, AttentiveNP
	# name = f'{pl_model_fn.__name__}_{pt_model_fn.__name__}'
	model = pl_model_fn(pt_model_fn, m_params, t_params, dm.fobs, splits)
	return model

def get_optmode(monitor):
	return {
		'val_loss': 'minimize',
		'val_reg_mae': 'minimize',
		'val_reg_mse': 'minimize',
		'val_binary_long_sharpe': 'maximize'
	}.get(monitor, 'maximize')

def get_callbacks(trial_dir, monitor='val_clf_accuracy'):
	mode = get_optmode(monitor)
	chk_callback = pl.callbacks.ModelCheckpoint(f'{trial_dir}chk{sep}',
		monitor=monitor, mode=mode)
	es_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=0,
		verbose=False, mode=mode)
	# ver_callbacks = (BatchNormVerificationCallback(),
			# BatchGradientVerificationCallback())
	callbacks = [chk_callback, es_callback]
	return callbacks

def get_trainer(trial_dir, callbacks, max_epochs, precision, gradient_clip_val=0):
	min_epochs = max(max_epochs // 4, 20)
	csv_log = pl.loggers.csv_logs.CSVLogger(trial_dir, name='', version='')
	# tb_log = pl.loggers.tensorboard.TensorBoardLogger(trial_dir, name='',
	# 	version='', log_graph=False)
	loggers = [csv_log]

	trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=min_epochs,
		logger=loggers, callbacks=callbacks, limit_val_batches=1.0,
		gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='norm',
		stochastic_weight_avg=False, auto_lr_find=False,
		amp_level='O1', precision=precision,
		default_root_dir=trial_dir, weights_summary=None,
		gpus=-1 if (torch.cuda.is_available()) else None)
	return trainer

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		exp_run(sys.argv[1:])

