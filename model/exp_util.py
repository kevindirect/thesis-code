"""
Kevin Patel
"""
import sys
import os
from os.path import sep, exists
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
# from verification.batch_norm import BatchNormVerificationCallback
# from verification.batch_gradient import BatchGradientVerificationCallback

from common_util import MODEL_DIR, NestedDefaultDict, load_json, rectify_json, dump_json, str_now, benchmark, makedir_if_not_exists, is_type, is_valid, isnt, get_cmd_args
from model.common import EXP_DIR


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
			if (name != 'epoch' or (len(ep := self.history['epoch'])==0 or ep[-1]!=val)):
				self.history[name].append(val)

	def log_hyperparams(self, params):
		pass

def modify_model_params(params_m, sm_name, model_name):
	if (sm_name == 'anp'):
		logging.info('modifying model params...')
		# Set deterministic/latent paths on/off depending on the model type.
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

def get_model(params_m, params_t, sm_name, model_name, dm, splits):
	modify_model_params(params_m, sm_name, model_name)
	if (sm_name in ('stcn', 'StackedTCN', 'GenericModel_StackedTCN')):
		from model.pl_generic import GenericModel
		from model.model_util import StackedTCN
		pl_model_fn, pt_model_fn = GenericModel, StackedTCN
	elif (sm_name in ('anp', 'AttentiveNP', 'NPModel_AttentiveNP')):
		from model.pl_np import NPModel
		from model.np_util import AttentiveNP
		pl_model_fn, pt_model_fn = NPModel, AttentiveNP
	model = pl_model_fn(pt_model_fn, params_m, params_t, dm.get_fshape(), splits)
	return model

def get_param_dir(sm_name, param_name):
	"""
	A valid parent model and set of hyperparameters.
	"""
	return EXP_DIR +sep.join([sm_name, param_name]) +sep

def get_study_dir(param_dir, model_name, data_name):
	"""
	A study is defined as a combination of treatment group (model), data, and hyperparameters
	"""
	return param_dir +sep.join([model_name, data_name]) +sep

def _get_trial_dir(study_dir, tid=None):
	"""
	A trial is a run of a study
	"""
	if (isnt(tid)):
		tid = str_now().replace(' ', '_').replace(':', '-')
	return study_dir +tid +sep

def get_trial_dir(param_dir, model_name, data_name, tid=None):
	"""
	A trial is a run of a study
	"""
	return _get_trial_dir(get_study_dir(param_dir, model_name, data_name), tid=tid)

# def dump_benchmarks(asset_name, dm):
# 	bench_data_name = f'{dm.interval[0]}_{dm.interval[1]}_{dm.ldata_name}'
# 	bench_dir = EXP_LOG_DIR +sep.join(['bench', asset_name, bench_data_name]) +sep
# 	if (not exists(bench_dir)):
# 		logging.info("dumping benchmarks...")
# 		makedir_if_not_exists(bench_dir)
# 		bench = dm.get_benchmarks()
# 		dm.dump_benchmarks_plots(bench, bench_dir)
# 		dm.dump_benchmarks_results(bench, bench_dir)
# 	else:
# 		logging.info("skipping benchmarks...")

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
		# monitor = 'val_clf_f1'
		monitor = 'val_conf_long_sharpe'
	elif (model_type == 'reg'):
		monitor = 'val_binary_long_sharpe'
	mode = get_optmode(monitor)[:3]

	chk_callback = ModelCheckpoint(dirpath=f'{trial_dir}chk{sep}',
		monitor=monitor, mode=mode)
	es_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=3,
		verbose=False, mode=mode)
	# es_callback = EarlyStopping(monitor='train_loss', verbose=False, mode='min', patience=0)
	# ver_callbacks = (BatchNormVerificationCallback(),
			# BatchGradientVerificationCallback())
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
		gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='norm',
		stochastic_weight_avg=False, auto_lr_find=False, precision=precision,
		accelerator="gpu", deterministic=det, #, amp_level='O1',
		default_root_dir=trial_dir, enable_model_summary=False,
		devices=-1 if (torch.cuda.is_available()) else None)
	return trainer

