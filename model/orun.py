"""
Kevin Patel
"""
import sys
import os
from os.path import sep, basename, dirname
from functools import partial
import logging

import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from common_util import MODEL_DIR, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args, compose, pd_split_ternary_to_binary, midx_intersect, pd_get_midx_level, pd_rows, df_midx_restack
from model.common import ASSETS, INTERVAL_YEARS, OPTUNA_DB_FNAME, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_HOURS, INTRADAY_LEN
from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs, get_hardcoded_feature_dfs, get_hardcoded_label_target_dfs


def optuna_run(argv):
	cmd_arg_list = ['dry-run', 'trials=', 'epochs=', 'run-hours=', 'model=', 'assets=', 'xdata=', 'ydata=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__), \
		script_pkg=basename(dirname(__file__)))
	dry_run = cmd_input['dry-run']
	n_trials = int(cmd_input['trials=']) if (cmd_input['trials=']) \
		else OPTUNA_N_TRIALS
	max_epochs = int(cmd_input['epochs=']) if (cmd_input['epochs=']) \
		else None
	hourly_timeout = int(cmd_input['run-hours=']) if (cmd_input['run-hours=']) \
		else OPTUNA_TIMEOUT_HOURS
	model_name = cmd_input['model='] or 'stcn'
	asset_name = cmd_input['assets='] or ASSETS[0]
	fdata_name = cmd_input['xdata='] or 'h_pba'
	ldata_name = cmd_input['ydata='] or 'ddir'
	data_name = f'{ldata_name}_{fdata_name}'
	fd = None

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

	# fdata options: d_rand, d_pba, d_vol, d_buzz, d_nonbuzz, h_rand, h_pba, h_vol, h_buzz
	if (fdata_name in ('d_rand', 'h_rand')):
		if (fdata_k[0] == 'd'):
			pass
		elif (fdata_k[0] == 'h'):
			print(INTRADAY_LEN)
			pass
	else:
		fd = get_xg_feature_dfs(asset_name, overwrite_cache=False)
		fdata = get_hardcoded_feature_dfs(fd, fdata_name)

	# ldata options: dcur (current period), ddir, ddir1, ddir2, dxfbdir1, dxbdir2
	if (ldata_name == 'dcur'):
		# Sanity check: 'Predict' the present ddir(t-1)
		fd = fd or get_xg_feature_dfs(asset_name)
		ldata = pd_split_ternary_to_binary(df_del_midx_level(\
			fd['d']['pba']['ddir']['pba_hoc_hdxret_ddir'] \
			.rename(columns={-1:'pba_hoc_hdxret_ddir'}), loc=1) \
			.replace(to_replace=-1, value=0).astype(int))
		tdata = pd_split_ternary_to_binary(df_del_midx_level(\
			fd['d']['pba']['dret']['pba_hoc_hdxret_dret'] \
			.rename(columns={-1:'pba_hoc_hdxret_dret'}), loc=1))
	else:
		ld, td = get_xg_label_target_dfs(asset_name, overwrite_cache=False)
		ldata, tdata = get_hardcoded_label_target_dfs(ld, td, ldata_name)

	# Set parameters of objective function to optimize:
	study_dir = MODEL_DIR +sep.join(['log', model_name, asset_name, data_name]) +sep
	study_name = ','.join([model_name, asset_name, data_name])
	study_db_path = f'sqlite:///{study_dir}{OPTUNA_DB_FNAME}'

	logging.info('cuda status: {}'.format( \
		'✓' if (torch.cuda.is_available()) else '🞩'))
	if (is_valid(max_epochs)):
		logging.info(f'max_epochs:  {max_epochs}')
	logging.info(f'num trials:  {n_trials} for {hourly_timeout} hour(s)')
	logging.info(f'study name:  {study_name}')
	logging.info(f'study dir:   {study_dir}')
	logging.info(f'study db:    {study_db_path}')

	if (dry_run):
		sys.exit()

	makedir_if_not_exists(study_dir)
	# logging.getLogger("lightning").setLevel(logging.ERROR) # Disable pl warnings
	torch.cuda.empty_cache()
	obj_fn = partial(objective, pl_model_fn=pl_model_fn, pt_model_fn=pt_model_fn,
		fdata=fdata, ldata=ldata, tdata=tdata, study_dir=study_dir,
		max_epochs=max_epochs)
	sampler = optuna.samplers.TPESampler(multivariate=True)
	study = optuna.create_study(storage=study_db_path, load_if_exists=True,
		sampler=sampler, direction='minimize', study_name=study_name)
	study.optimize(obj_fn, n_trials=n_trials, timeout=hourly_timeout*60*60,
		n_jobs=1, gc_after_trial=False, show_progress_bar=False)
	# TODO save/record random seed used


def objective(trial, pl_model_fn, pt_model_fn, fdata, ldata, tdata,
	study_dir, max_epochs=None):
	"""
	Args:
		trial
		pl_model_fn: pytorch lightning model wrapper module constructor
		pt_model_fn: pytorch model module constructor
		max_epochs: override traning params epoch value
	"""
	trial_dir = f'{study_dir}{str(trial.number).zfill(6)}{sep}'
	logging.info(f'trial dir:   {trial_dir}')
	data = common_interval_data(fdata, ldata, tdata)
	t_params = pl_model_fn.suggest_params(trial, num_classes=2)
	m_params = pt_model_fn.suggest_params(trial, num_classes=2, add_ob=True)
	mdl = pl_model_fn(pt_model_fn, m_params, t_params, data)
	# logging.info(f'gpu mem: {torch.cuda.max_memory_allocated()} mb')

	csv_log = pl.loggers.csv_logs.CSVLogger(trial_dir, name='', version='')
	tb_log = pl.loggers.tensorboard.TensorBoardLogger(trial_dir, name='', \
		version='', log_graph=False)
	checkpoint_callback = pl.callbacks.ModelCheckpoint(f'{trial_dir}chk{sep}', \
		monitor='val_loss', mode='min')
	es_callback = t_params['prune_trials'] and \
		PyTorchLightningPruningCallback(trial, monitor='val_loss')

	trainer = pl.Trainer(max_epochs=max_epochs or t_params['epochs'],
			logger=[csv_log, tb_log], callbacks=[es_callback],
			checkpoint_callback=checkpoint_callback,
			limit_val_batches=1.0, gradient_clip_val=0., track_grad_norm=2,
			auto_lr_find=False, amp_level='O1', precision=16,
			default_root_dir=trial_dir, weights_summary=None,
			gpus=-1 if (torch.cuda.is_available()) else None)
	trainer.fit(mdl)
	pl_model_fn.fix_metrics_csv('metrics.csv', dir_path=trial_dir)

	return trainer.callback_metrics['val_loss']


def common_interval_data(fdata, ldata, tdata, interval=INTERVAL_YEARS):
	"""
	Intersect common data over interval and return it
	"""
	com_idx = midx_intersect(pd_get_midx_level(fdata), pd_get_midx_level(ldata), \
		pd_get_midx_level(tdata))
	com_idx = com_idx[(com_idx > interval[0]) & (com_idx < interval[1])]
	feature_df, label_df, target_df = map(compose(partial(pd_rows, idx=com_idx), \
		df_midx_restack), [fdata, ldata, tdata])
	assert(all(feature_df.index.levels[0]==label_df.index.levels[0]))
	assert(all(feature_df.index.levels[0]==target_df.index.levels[0]))
	return feature_df, label_df, target_df


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		optuna_run(sys.argv[1:])

