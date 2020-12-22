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

from common_util import MODEL_DIR, rectify_json, dump_json, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args
from model.common import ASSETS, INTERVAL_YEARS, WIN_SIZE, OPTUNA_DB_FNAME, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_HOURS, INTRADAY_LEN
from model.pl_xgdm import XGDataModule


def optuna_run(argv):
	cmd_arg_list = ['dry-run', 'trials=', 'epochs=', 'run-hours=', 'model=', 'assets=', 'xdata=', 'ydata=', 'optuna-monitor=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__), \
		script_pkg=basename(dirname(__file__)))
	dry_run = cmd_input['dry-run']
	n_trials = int(cmd_input['trials=']) if (cmd_input['trials=']) \
		else OPTUNA_N_TRIALS
	max_epochs = int(cmd_input['epochs=']) if (cmd_input['epochs=']) \
		else None
	min_epochs = 20
	hourly_timeout = int(cmd_input['run-hours=']) if (cmd_input['run-hours=']) \
		else OPTUNA_TIMEOUT_HOURS
	model_name = cmd_input['model='] or 'stcn'
	asset_name = cmd_input['assets='] or ASSETS[0]
	fdata_name = cmd_input['xdata='] or 'h_pba_mzo,h_vol_mzo'
	ldata_name = cmd_input['ydata='] or 'ddir'
	monitor = cmd_input['optuna-monitor='] or 'val_loss'
	optimize_dir = {
		'val_loss': 'minimize'
	}.get(monitor, 'maximize')

	# Init the datamodule
	dm = XGDataModule({'window_size': WIN_SIZE}, asset_name, fdata_name, ldata_name,
		interval=INTERVAL_YEARS, overwrite_cache=False)
	dm.prepare_data()
	dm.setup()

	# model options: stcn, anp
	if (model_name in ('stcn', 'StackedTCN', 'GenericModel_StackedTCN')):
		from model.pl_generic import GenericModel
		from model.model_util import StackedTCN
		pl_model_fn, pt_model_fn = GenericModel, StackedTCN
	elif (model_name in ('anp', 'AttentiveNP', 'NPModel_AttentiveNP')):
		from model.pl_np import NPModel
		from model.np_util import AttentiveNP
		pl_model_fn, pt_model_fn = NPModel, AttentiveNP
	model_name = f'{pl_model_fn.__name__}_{pt_model_fn.__name__}'

	# Set parameters of objective function to optimize:
	study_dir = MODEL_DIR \
		+sep.join(['olog', model_name, asset_name, dm.name, monitor]) +sep
	study_name = ','.join([model_name, asset_name, dm.name, monitor])
	study_db_path = f'sqlite:///{study_dir}{OPTUNA_DB_FNAME}'

	logging.debug('cuda status: {}'.format( \
		'✓' if (torch.cuda.is_available()) else '🞩'))
	if (is_valid(max_epochs)):
		logging.info(f'max_epochs:  {max_epochs}')
	logging.info(f'num trials:  {n_trials} for {hourly_timeout} hour(s)')
	logging.info(f'study name:  {study_name}')
	logging.debug(f'study dir:   {study_dir}')
	logging.debug(f'study db:    {study_db_path}')

	if (dry_run):
		sys.exit()

	makedir_if_not_exists(study_dir)
	bench_fname = 'benchmark.json'
	if (not exists('{study_dir}{bench_fname}')):
		dump_json(dm.get_benchmarks(), bench_fname, study_dir)
	# logging.getLogger("lightning").setLevel(logging.ERROR) # Disable pl warnings
	torch.cuda.empty_cache()
	obj_fn = partial(objective, pl_model_fn=pl_model_fn, pt_model_fn=pt_model_fn,
		dm=dm, monitor=monitor, direction=optimize_dir, study_dir=study_dir,
		max_epochs=max_epochs, min_epochs=min_epochs)
	sampler = optuna.samplers.TPESampler(multivariate=True)
	pruner = PercentilePruner(percentile=50.0, n_startup_trials=10, \
		n_warmup_steps=min_epochs, interval_steps=10) # Top percentile of trials are kept
	study = optuna.create_study(storage=study_db_path, load_if_exists=True,
		sampler=sampler, pruner=pruner, direction=optimize_dir, study_name=study_name)
	study.optimize(obj_fn, n_trials=n_trials, timeout=hourly_timeout*60*60,
		catch=(), n_jobs=1, gc_after_trial=False, show_progress_bar=False)
	# TODO save/record random seed used


def objective(trial, pl_model_fn, pt_model_fn, dm, monitor, direction,
	study_dir, max_epochs=None, min_epochs=1):
	"""
	Args:
		trial
		pl_model_fn: pytorch lightning model wrapper module constructor
		pt_model_fn: pytorch model module constructor
		dm (pl.DataModule): data
		monitor: the validation metric optina checkpoints and monitors on
		max_epochs: override traning params epoch value
	"""
	trial_dir = f'{study_dir}{str(trial.number).zfill(6)}{sep}'
	makedir_if_not_exists(trial_dir)
	logging.debug(f'trial dir:   {trial_dir}')

	m_params = pt_model_fn.suggest_params(trial, num_classes=2, add_ob=True)
	t_params = pl_model_fn.suggest_params(trial, num_classes=2)
	dm.update_params(t_params)
	mdl = pl_model_fn(pt_model_fn, m_params, t_params, dm.fobs)

	dump_json(rectify_json(m_params), 'params_m.json', trial_dir)
	dump_json(rectify_json(t_params), 'params_t.json', trial_dir)
	logging.debug(f'gpu mem (mb): {torch.cuda.max_memory_allocated()}')
	logging.debug(f'model params: {m_params}')
	logging.debug(f'train params: {t_params}')

	csv_log = pl.loggers.csv_logs.CSVLogger(trial_dir, name='', version='')
	tb_log = pl.loggers.tensorboard.TensorBoardLogger(trial_dir, name='', \
		version='', log_graph=False)
	chk_callback = pl.callbacks.ModelCheckpoint(f'{trial_dir}chk{sep}', \
		monitor=monitor, mode=direction[:3])
	es_callback = PyTorchLightningPruningCallback(trial, \
		monitor=monitor) # hook for optuna pruner
	ver_callbacks = (BatchNormVerificationCallback(), \
		BatchGradientVerificationCallback())

	trainer = pl.Trainer(max_epochs=max_epochs or t_params['epochs'],
			min_epochs=min_epochs, logger=[csv_log, tb_log],
			callbacks=[chk_callback, es_callback, *ver_callbacks],
			limit_val_batches=1.0, gradient_clip_val=0., #track_grad_norm=2,
			auto_lr_find=False, amp_level='O1', precision=16,
			default_root_dir=trial_dir, weights_summary=None,
			gpus=-1 if (torch.cuda.is_available()) else None)
	trainer.fit(mdl, datamodule=dm)
	# pl_model_fn.fix_metrics_csv('metrics.csv', dir_path=trial_dir)

	return trainer.callback_metrics[monitor]


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		optuna_run(sys.argv[1:])

