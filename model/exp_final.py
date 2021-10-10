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
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import loggers as pl_loggers
# from verification.batch_norm import BatchNormVerificationCallback
# from verification.batch_gradient import BatchGradientVerificationCallback

from common_util import MODEL_DIR, load_df, deep_update, rectify_json, load_json, dump_json, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args
from model.common import ASSETS, INTERVAL_YEARS, WIN_SIZE, INTRADAY_LEN, EXP_LOG_DIR, EXP_PARAMS_DIR
from model.pl_xgdm import XGDataModule
from model.exp_run import dump_benchmarks, get_model, get_optmode, get_trainer


def exp_final_run(argv):
	cmd_arg_list = ['dry-run', 'intstart=', 'assets=', 'smodel=', 'models=', 'xdata=', 'ydata=']
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
	dm_name = XGDataModule.get_name(int_years, fdata_name, ldata_name)

	# model args
	sm_name = cmd_input['smodel='] or 'anp'
	if (is_valid(model_names := cmd_input['models='])):
		model_names = model_names.split(',')
	elif (sm_name == 'anp'):
		model_names = ['base', 'cnp', 'lnp', 'np']

	MAX_EPOCHS = 100
	exp_params_dir = EXP_PARAMS_DIR +sep.join(['final', sm_name]) +sep
	exp_dir = EXP_LOG_DIR +sep.join(['final', sm_name]) +sep
	splits = ('train', 'val', 'test')

	logging.info(f'{int_years=}')
	logging.info(f'{fdata_name=}')
	logging.info(f'{ldata_name=}')
	logging.info(f'{asset_names=}')
	logging.info(f'model: {sm_name}->{model_names}')
	logging.debug('cuda: {}'.format('✓' if (torch.cuda.is_available()) else '🞩'))

	for asset_name in asset_names:
		logging.info(f'{asset_name=}')

		for model_name in model_names:
			logging.info(f'{model_name=}')
			model_path = sep.join([asset_name, model_name, dm_name]) +sep

			for study_num in os.listdir(model_params_dir := f'{exp_params_dir}{model_path}'):
				logging.info(f'{study_num=}')
				study_path = f'{model_path}{study_num}{sep}'

				for trial_num in os.listdir(study_params_dir := f'{exp_params_dir}{study_path}'):
					trial_path = f'{study_path}{trial_num}{sep}'
					trial_params_dir = f'{exp_params_dir}{trial_path}'
					trial_dir = f'{exp_dir}{trial_path}'

					logging.info('loading params...')
					t_params = load_json('params_t.json', trial_params_dir)
					m_params = load_json('params_m.json', trial_params_dir)
					assert_valid_model(model_name, m_params)

					logging.info('loading data...')
					dm = XGDataModule(t_params, asset_name,
						fdata_name, ldata_name, interval=int_years,
						fret=None, overwrite_cache=False)
					dm.prepare_data()
					dm.setup()
					dump_benchmarks(asset_name, dm)

					if (dry_run):
						# print(f'{t_params=}')
						# print(f'{m_params=}')
						logging.info('dry-run: continuing study loop')
						continue

					logging.info('dump metadata...')
					makedir_if_not_exists(trial_dir)
					dump_json(rectify_json(m_params), 'params_m.json', trial_dir)
					dump_json(rectify_json(t_params), 'params_t.json', trial_dir)

					# Build and train model
					model = get_model(sm_name, m_params, t_params, dm, splits)
					callbacks = get_fcallbacks(trial_dir)
					trainer = get_ftrainer(trial_dir, callbacks,
						t_params['epochs'], MAX_EPOCHS,
						model.get_precision())
					trainer.fit(model, datamodule=dm)

					if ('test' in splits):
						trainer.test(model, datamodule=dm, verbose=False)

					# Dump metadata and results
					model.dump_plots(trial_dir, model_name, dm)
					model.dump_results(trial_dir, model_name)

		torch.cuda.empty_cache()

def assert_valid_model(model_name, m_params):
	if (model_name == 'base'):
		assert not m_params['use_det_path'] and not m_params['use_lat_path']
	elif (model_name == 'cnp'):
		assert m_params['use_det_path'] and not m_params['use_lat_path']
	elif (model_name == 'lnp'):
		assert not m_params['use_det_path'] and m_params['use_lat_path']
	elif (model_name == 'np'):
		assert m_params['use_det_path'] and m_params['use_lat_path']

def get_fcallbacks(trial_dir, monitor='val_clf_accuracy'):
	"""
	load model with checkpointed weights:
		model = MyLightingModule.load_from_checkpoint(f'{trial_dir}chk{sep}{name}')
	"""
	mode = get_optmode(monitor)[:3]
	chk_callback = ModelCheckpoint(dirpath=f'{trial_dir}chk{sep}',
		monitor=monitor, mode=mode)
	es_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=0,
		verbose=False, mode=mode)
	# ver_callbacks = (BatchNormVerificationCallback(),
			# BatchGradientVerificationCallback())
	callbacks = [chk_callback, es_callback]
	return callbacks

def get_ftrainer(trial_dir, callbacks, min_epochs, max_epochs, precision, gradient_clip_val=0):
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
		exp_final_run(sys.argv[1:])

