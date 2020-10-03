"""
Kevin Patel
"""
import sys
import os
from os.path import sep, basename
from functools import partial
import logging

import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import optuna

from common_util import MODEL_DIR, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args, compose, pd_split_ternary_to_binary, midx_intersect, pd_get_midx_level, pd_rows, df_midx_restack
from model.common import INTRADAY_LEN
from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs, get_hardcoded_feature_dfs, get_hardcoded_label_target_dfs


ASSETS = ('sp_500', 'russell_2000', 'nasdaq_100', 'dow_jones')
INTERVAL_YEARS = ('2009', '2018')
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 600

def optuna_run(argv):
	cmd_arg_list = ['dry-run', 'assets=', 'model=', 'xdata=', 'ydata=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	dry_run = cmd_input['dry-run']
	asset_name = cmd_input['assets='] or ASSETS[0]
	model_name = cmd_input['model='] or 'stcn'
	fdata_name = cmd_input['xdata='] or 'h_pba'
	ldata_name = cmd_input['ydata='] or 'ddir'
	data_name = f"{ldata_name}_{fdata_name}"
	fd = None

	# Set model:
	if (model_name in ('stcn', 'StackedTCN', 'GenericModel_StackedTCN')):
		from model.pl_generic import GenericModel
		from model.model_util import StackedTCN
		pl_model_fn, pt_model_fn = GenericModel, StackedTCN
	elif (model_name in ('anp', 'AttentiveNP', 'NPModel_AttentiveNP')):
		from model.pl_np import NPModel
		from model.np_util import AttentiveNP
		pl_model_fn, pt_model_fn = NPModel, AttentiveNP
	model_name = f"{pl_model_fn.__name__}_{pt_model_fn.__name__}"

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
	obj_fn = partial(objective, pl_model_fn=pl_model_fn, pt_model_fn=pt_model_fn,
		fdata=fdata, ldata=ldata, tdata=tdata, study_dir=study_dir)

	logging.info('cuda status: {}'.format( \
		'✓' if (torch.cuda.is_available()) else '🞩'))
	logging.info(f'study name:  {study_name}')
	logging.info(f'study dir:   {study_dir}')

	if (dry_run):
		sys.exit()

	study = optuna.create_study(study_name=study_name, direction='minimize')
	study.optimize(obj_fn, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)

	# TODO dump study results
	# print("Number of finished trials: {}".format(len(study.trials)))
	# print("Best trial:")
	# trial = study.best_trial
	# print("		Value: {}".format(trial.value))
	# print("		Params: ")
	# for key, value in trial.params.items():
	# 	print("			{}: {}".format(key, value))

def objective(trial, pl_model_fn, pt_model_fn, fdata, ldata, tdata, study_dir):
	"""
	Args:
		pl_model_fn: pytorch lightning model wrapper module constructor
		pt_model_fn: pytorch model module constructor
	"""
	data = common_interval_data(fdata, ldata, tdata)
	t_params = pl_model_fn.suggest_params(trial)
	m_params = pt_model_fn.suggest_params(trial)
	mdl = pl_model_fn(pt_model_fn, m_params, t_params, data)
	t_params['epochs'] = 1 # XXX
	
	# checkpoint_callback = pl.callbacks.ModelCheckpoint(
	# 	study_dir +sep.join(['version_{}'.format(trial.number), 'chk']),
	# 	monitor='val_loss', mode='min'
	# )

	# escb = PyTorchLightningPruningCallback(trial, monitor='val_loss') if (prune) else
	# 	EarlyStopping(monitor='val_loss', min_delta=0.00, patience=30, verbose=False, \
	# 		mode='min')
	tb_logger = pl.loggers.tensorboard.TensorBoardLogger(study_dir, name='tb')
	csv_logger = pl.loggers.csv_logs.CSVLogger(study_dir, name='csv')
	trainer = pl.Trainer(max_epochs=t_params['epochs'], logger=[tb_logger, csv_logger],
			# checkpoint_callback=checkpoint_callback,
			limit_val_batches=1.0, gradient_clip_val=0., track_grad_norm=2,
			auto_lr_find=False, amp_level='O1', precision=16,
			default_root_dir=study_dir, weights_summary='top',
			gpus=-1 if (torch.cuda.is_available()) else None)
	trainer.fit(mdl)
	print(mdl.logger.metrics[-1])
	sys.exit()
	# TODO save/load model checkpoints
	# TODO get val_loss and return
	return None

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


# def x_main(trial, pl_model_fn, name, train=True, prune=True, pct_test=0.5):
# 	# PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
# 	# filenames match. Therefore, the filenames for each trial must be made unique.

# 	checkpoint_callback = pl.callbacks.ModelCheckpoint(
# 		sep.join([LL_DIR, name, 'version_{}'.format(trial.number), 'chk']),
# 		monitor='val_loss', mode='min'
# 	)
# 	# hparams = dict(**trial.params, **trial.user_attrs)

# 	trainer = pl.Trainer(
# 		logger=logger,
# 		val_percent_check=pct_test,
# 		checkpoint_callback=checkpoint_callback,
# 		max_epochs=hparams['epochs'],
# 		weights_summary='top',
# 		gpus=-1 if (torch.cuda.is_available()) else None,
# 		early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='val_loss')
# 		if (prune)
# 		else EarlyStopping(patience=hparams['patience'] * 2, monitor='val_loss', \
# 			verbose=True)
# 	)

# 	model = pl_model_fn(hparams)
# 	if (train):
# 		trainer.fit(model)
# 	return model, trainer


def x_objective(trial, pl_model_fn, name, user_attrs):
	"""
	For optuna hparam opt.
	"""
	trial = pl_model_fn.add_suggest(trial)
	[trial.set_user_attr(k, v) for k, v in user_attrs.items()]

	logger.debug(dict(number=trial.number, params=trial.params, user_attrs=trial.user_attrs))
	model, trainer = x_main(trial, pl_model_fn, name=name)

	# Load checkpoint
	checkpoints = sorted(Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt"))
	if len(checkpoints):
		checkpoint = checkpoints[-1]
		device = next(model.parameters()).device
		logger.info(f"Loading checkpoint {checkpoint}")
		model = model.load_from_checkpoint(checkpoint).to(device)

	trainer.test(model)
	logger.info("logger.metrics {}".format(model.logger.metrics[-1:]))
	model.logger.experiment.add_hparams(trial.params, model.logger.metrics[-1])
	model.logger.save()

	return model.logger.metrics[-1]["agg_test_score"]


# def add_number(trial, model_dir):
# 	# For manual experiment we will start at -1 and deincr by 1
# 	versions = [int(s.stem.split("_")[-1]) for s in model_dir.glob("version_*")] + [-1]
# 	trial.number = min(versions) - 1
# 	# logger.debug("trial.number", trial.number)
# 	return trial


# def run_trial(name, pl_model_fn, params, user_attrs, LL_DIR,
# 	plot_from_loader=plot_from_loader, number=None):
# 	logger.info(f"now run `tensorboard --logdir {LL_DIR}`")
# 	makedir_if_not_exists(sep.join([LL_DIR, name]).mkdir(parents=True, exist_ok=True))

# 	if (getattr(pl_model_fn, 'DEFAULT_ARGS', None)):
# 		# add default args
# 		params = {**pl_model_fn.DEFAULT_ARGS, **params}
# 	else:
# 		logger.warning(f"No default args on {pl_model_fn}")

# 	# Make trial
# 	trial = optuna.trial.FixedTrial(params=params)
# 	trial = pl_model_fn.add_suggest(trial)

# 	if (isnt(number)):
# 		trial = add_number(trial, LL_DIR / name)
# 	else:
# 		trial.number = number

# 	# Add user attributes
# 	[trial.set_user_attr(k, v) for k, v in user_attrs.items()]
# 	model, trainer = main(
# 		trial, pl_model_fn, name=name, LL_DIR=LL_DIR, train=False, prune=False
# 	)
# 	logger.info('trial number=%s name=%s, trial=%s params=%s attrs=%s', trial.number, trainer.logger.name, trial, trial.params, trial.user_attrs)

# 	checkpoints = sorted(Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt"))
# 	if len(checkpoints)==0 or number is None:
# 		try:
# 			trainer.fit(model)
# 		except KeyboardInterrupt:
# 			logger.warning('KeyboardInterrupt, skipping rest of training')
# 			pass

# 		# Plot
# 		loader = model.val_dataloader()
# 		dset_test = loader.dataset
# 		label_names = dset_test.label_names
# 		plot_from_loader(model.val_dataloader(), model, i=670, title='overfit val 670')
# 		plt.show()
# 		plot_from_loader(model.train_dataloader(), model, i=670, title='overfit train 670')
# 		plt.show()
# 		plot_from_loader(model.test_dataloader(), model, i=670, title='overfit test 670')
# 		plt.show()

# 	# Load checkpoint
# 	checkpoints = sorted(Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt"))
# 	if len(checkpoints):
# 		checkpoint = checkpoints[-1]
# 		device = next(model.parameters()).device
# 		logger.info(f"Loading checkpoint {checkpoint}")
# 		model = model.load_from_checkpoint(checkpoint).to(device)

# 		# Plot
# 		plot_from_loader(model.val_dataloader(), model, i=670, title='val 670')
# 		plt.show()
# 		plot_from_loader(model.train_dataloader(), model, i=670, title='train 670')
# 		plt.show()
# 		plot_from_loader(model.test_dataloader(), model, i=670, title='test 670')
# 		plt.show()
# 	else:
# 		logger.warning('no checkpoints')

# 	try:
# 		trainer.test(model)
# 	except KeyboardInterrupt:
# 		logger.warning('KeyboardInterrupt, skipping rest of testing')
# 		pass
# 	return trial, trainer, model


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		optuna_run(sys.argv[1:])

