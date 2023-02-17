import sys
import os
from os.path import sep, exists
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
# from verification.batch_norm import BatchNormVerificationCallback
# from verification.batch_gradient import BatchGradientVerificationCallback

from common_util import MODEL_DIR, deep_update, load_json, rectify_json, dump_json, load_df, dump_df, benchmark, dt_now, makedir_if_not_exists, is_type, is_valid, isnt
from model.common import EXP_DIR, MAX_EPOCHS
from model.viz import dump_fig, plot_df_line, plot_df_scatter, plot_df_line_subplot, plot_df_scatter_subplot, plot_df_hist_subplot


class MemLogger(LightningLoggerBase):
	"""
	"Logs" in memory, used to access computed metrics at runtime.

	Modified from: https://stackoverflow.com/questions/69276961/how-to-extract-loss-and-accuracy-from-logger-by-each-epoch-in-pytorch-lightning
	"""
	def __init__(self):
		super().__init__()
		self.history = defaultdict(list) 

	@property
	def name(self):
		return "MemLogger"

	@property
	def version(self):
		return "1.0"

	@property
	@rank_zero_experiment
	def experiment(self):
		pass

	@rank_zero_only
	def log_metrics(self, metrics, step):
		for name, val in metrics.items():
			if (name != "epoch" or (len(ep := self.history["epoch"])==0 or ep[-1]!=val)):
				self.history[name].append(val)

	def log_hyperparams(self, params):
		pass

	def history_df(self):
		d = {k: v for k, v in self.history.items() if (not k.startswith('test_'))}
		return pd.DataFrame.from_dict(d, orient="columns").set_index("epoch")


def modify_model_params(params_m, sm_name, model_name):
	if (sm_name == 'anp'):
		logging.info('modifying model params...')
		# Switch deterministic/latent paths on/off depending on the model type
		if (model_name == 'base'):
			params_m['use_det_path'] = False
			params_m['use_lat_path'] = False
		elif (model_name == 'cnp'):
			params_m['use_det_path'] = True
			params_m['use_lat_path'] = False
		elif (model_name == 'lnp'):
			params_m['use_det_path'] = False
			params_m['use_lat_path'] = True
		elif (model_name == 'np'):
			params_m['use_det_path'] = True
			params_m['use_lat_path'] = True

def get_model(params_m, params_d, sm_name, model_name, splits, dm):
	modify_model_params(params_m, sm_name, model_name)
	if (sm_name in ('stcn', 'StackedTCN', 'GenericModel_StackedTCN')):
		from model.pl_generic import GenericModel
		from model.model_util import StackedTCN
		pl_model_fn, pt_model_fn = GenericModel, StackedTCN
	elif (sm_name in ('anp', 'AttentiveNP', 'NPModel_AttentiveNP')):
		from model.pl_np import NPModel
		from model.np_util import AttentiveNP
		pl_model_fn, pt_model_fn = NPModel, AttentiveNP
	model = pl_model_fn(pt_model_fn, params_m, params_d, dm.get_fshape(), splits)
	return model

def get_param_dir(sm_name, param_name, dir_path=EXP_DIR):
	"""
	A valid parent model and set of hyperparameters.
	"""
	return dir_path +sep.join([sm_name, param_name]) +sep

def get_study_dir(param_dir, model_name, data_name):
	"""
	A study is defined as a combination of treatment group (model), data, and hyperparameters
"""
	return param_dir +sep.join([model_name, data_name]) +sep

def get_study_name(study_dir, dir_path=EXP_DIR):
	return study_dir.replace(dir_path, '').rstrip(sep).replace(sep, ',')

def get_trial_dir(study_dir, trial_id):
	"""
	A trial is a run of a study
	"""
	return study_dir +trial_id +sep

def get_optmode(monitor):
	return {
		'val_loss': 'minimize',
		'val_reg_mae': 'minimize',
		'val_reg_mse': 'minimize',
		'val_binary_long_sharpe': 'maximize'
	}.get(monitor, 'maximize')

def get_callbacks(trial_dir, model_type):
	"""
	load model with checkpointed weights:
		model = MyLightingModule.load_from_checkpoint(f'{trial_dir}chk{sep}{name}')
	"""
	if (model_type == 'clf'):
		monitor = 'val_clf_f1'
	elif (model_type == 'reg'):
		monitor = 'val_reg_mse'
	mode = get_optmode(monitor)[:3]

	chk_callback = ModelCheckpoint(dirpath=f'{trial_dir}chk{sep}',
		monitor=monitor, mode=mode)
	es_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=3,
		verbose=False, mode=mode)
	# es_callback = EarlyStopping(monitor='train_loss', verbose=False, mode='min', patience=0)
	# ver_callbacks = (BatchNormVerificationCallback(), BatchGradientVerificationCallback())
	callbacks = [chk_callback, es_callback]
	return callbacks

def get_trainer(trial_dir, callbacks, min_epochs, max_epochs, precision, plseed=None, gradient_clip_val=2):
	mem_log = MemLogger()
	csv_log = pl.loggers.csv_logs.CSVLogger(trial_dir, name='', version='')
	# tb_log = pl.loggers.tensorboard.TensorBoardLogger(trial_dir, name='', version='', log_graph=True)
	loggers = [mem_log, csv_log]

	if (is_valid(plseed)):
		pl.utilities.seed.seed_everything(plseed)
	det = False

	trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=min_epochs,
		logger=loggers, callbacks=callbacks, limit_val_batches=1.0,
		# gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='norm',
		stochastic_weight_avg=False, auto_lr_find=False, precision=precision,
		accelerator="gpu", deterministic=det, #, amp_level='O1',
		default_root_dir=trial_dir, enable_model_summary=False,
		# track_grad_norm=2,
		# detect_anomaly=True,
		devices=-1 if (torch.cuda.is_available()) else None)
	return trainer

def dump_plot_metric(df, dir_path, metric, splits, title, fname,
	linestyles=["solid", "dashed", "dotted"]):
	plot_df_line(df.loc[:, [f"{s}_{metric}" for s in splits]],
		title=title, xlabel="epochs", ylabel="loss", linestyles=linestyles)
	dump_fig(f"{dir_path}{fname}", transparent=False)

def dump_plot_pred(df, dir_path, ylabel, title, fname,
	linestyles=["solid", "dashed", "dotted"]):
	plot_df_line(df.loc[:, ["yt", "pred_mean"]],
		title=title, xlabel="date", ylabel=ylabel,
		linestyles=linestyles)
	plt.fill_between(
		df.index,
		df["pred_mean"] - df["pred_std"],
		df["pred_mean"] + df["pred_std"],
		alpha=.25,
		facecolor="orange",
		interpolate=True,
		label="σ",
	)
	plt.fill_between(
		df.index,
		df["pred_mean"] - df["pred_std"]*2,
		df["pred_mean"] + df["pred_std"]*2,
		alpha=.125,
		facecolor="orange",
		interpolate=True,
		label="2σ",
	)
	yexpand = .05 * (df["yt"].max() - df["yt"].min())
	plt.ylim([df["yt"].min()-yexpand, df["yt"].max()+yexpand])
	dump_fig(f"{dir_path}{fname}", transparent=False)

def dump_plot_pred_scatter(df, dir_path, ylabel, title, fname, alpha=.3):
	plot_df_scatter(df.loc[:, ["yt", "pred_mean"]],
		title=title, xlabel="date", ylabel=ylabel, alpha=alpha)
	plt.errorbar(df.index, df["pred_mean"], yerr=df["pred_std"])
	dump_fig(f"{dir_path}{fname}", transparent=False)

def fix_metrics_csv(dir_path, fname="metrics"):
	"""
	Fix Pytorch Lightning v10 logging rows in the same epoch on
	separate rows.
	"""
	csv_df = load_df(fname, dir_path=dir_path, data_format="csv")
	test_cols = [col for col in csv_df.columns if col.startswith('test_')]
	if (len(test_cols) > 0):
		dump_df(csv_df[test_cols].dropna(how="any"), f"{fname}_test", dir_path=dir_path, data_format="csv")
	csv_df = csv_df.loc[:, ~csv_df.columns.isin(test_cols)].groupby("epoch").ffill().dropna(how="any")
	os.rename(f"{dir_path}{fname}.csv", f"{dir_path}{fname}.old.csv")
	dump_df(csv_df, f"{fname}", dir_path=dir_path, data_format="csv")
	logging.debug(f"fixed {fname}")
	return csv_df

def run_exp(study_dir, params_m, params_d, sm_name, model_name, splits, dm, max_epochs=MAX_EPOCHS, seed=None):
	seed = seed or dt_now().timestamp()
	trial_dir = get_trial_dir(study_dir, str(seed))
	makedir_if_not_exists(trial_dir)

	model = get_model(params_m, params_d, sm_name, model_name, splits, dm)
	callbacks = get_callbacks(trial_dir, model.model_type)
	trainer = get_trainer(trial_dir, callbacks, params_m['epochs'], max_epochs,
		model.precision, seed)
	# logging.debug(f'gpu mem (mb): {torch.cuda.max_memory_allocated()}')
	trainer.fit(model, datamodule=dm)
	if ('test' in splits):
		trainer.test(model, datamodule=dm, verbose=False)
	return trial_dir, model, trainer

def dump_exp(trial_dir, params_m, params_d, sm_name, model_name, splits, dm, model, trainer, metrics=["loss", "reg_mse", "reg_mae"]):
	# Dump prediction plots
	dfs_pred = {split: model.pred_df(dm.get_dataloader(split), dm.index[split]) for split in splits}
	for split, df_pred in dfs_pred.items():
		dump_df(df_pred, f"{split}_pred", trial_dir, "csv")
		dump_plot_pred(df_pred, trial_dir,
			dm.target_name,
			f"{dm.asset_name} {sm_name}_{model_name} {split} {dm.target_name}".lower(),
			f"plot_{split}_pred")

	# Dump params, results, and metrics over train / val
	df_hist = fix_metrics_csv(trial_dir)
	dump_json(params_d, "params_d.json", dir_path=trial_dir)
	dump_json(params_m, "params_m.json", dir_path=trial_dir)
	# df_hist = trainer.logger[0].history_df()
	for metric in metrics:
		dump_plot_metric(df_hist, trial_dir, metric, tuple(filter(lambda s: s!="test", splits)),
			f"{dm.asset_name} {sm_name}_{model_name} {metric}".lower(),
			f"plot_{metric}")
	return df_hist

def run_dump_exp(study_dir, params_m, params_d, sm_name, model_name, splits, dm, seed=None):
	"""
	Wraps around run_exp()->dump_exp() calls
	"""
	trial_dir, model, trainer = run_exp(study_dir, params_m, params_d,
		sm_name, model_name, splits, dm, seed=seed
	)
	df_hist = dump_exp(trial_dir, params_m, params_d, sm_name, model_name,
		splits, dm, model, trainer, metrics=["loss", "reg_mse", "reg_mae"]
	)
	return trial_dir, model, trainer, df_hist

def get_objective_fn(study_dir, params_m, params_d, sm_name, model_name, splits, dm, obj, suggestor_m):
	"""
	Returns optuna objective function that wraps run_dump_exp() 
	"""
	def objective_fn(trial):
		# trial_num = str(trial.number).zfill(6)
		deep_update(params_m, suggestor_m(trial))
		hist_df = run_dump_exp(study_dir, params_m, params_d, sm_name, model_name,
			splits, dm)[-1]
		return hist_df[obj].iloc[-1]

	return objective_fn
