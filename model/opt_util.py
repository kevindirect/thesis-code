"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
# from verification.batch_norm import BatchNormVerificationCallback
# from verification.batch_gradient import BatchGradientVerificationCallback
import optuna
from optuna.pruners import PercentilePruner, HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback

from common_util import MODEL_DIR, is_type, is_valid, isnt
from model.common import ASSETS, INTERVAL_YEARS, WIN_SIZE, OPTUNA_DB_FNAME, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_HOURS, INTRADAY_LEN, EXP_LOG_DIR, EXP_PARAMS_DIR
# from model.pl_xgdm import XGDataModule
# from model.exp_run import dump_benchmarks, get_train_params, get_model_params, get_optmode, get_trial_dir, get_model, get_trainer


def get_suggest_train(base, sm_name, model_name, asset_name, dm, sample_out=False):
	"""
	Return a train param suggest function to overwrite default hyperparameters during
	an optuna trial.
	"""
	def suggestor(trial):
		# window_size = trial.suggest_int('window_size', 5*1, 5*4, step=5) #4
		batch_size = trial.suggest_int('batch_size', 64, 128, step=64) #2
		batch_step_size = batch_size//2
		train_resample = batch_size//4
		train_target_overlap = batch_size//8
		model_type = base['loss'].split('-')[0]

		return {
			# 'window_size': window_size,
			'batch_size': batch_size,
			'batch_step_size': batch_step_size,
			'context_size': batch_step_size,
			'target_size': batch_step_size,
			'train_resample': train_resample,
			'train_target_overlap': train_target_overlap,
			'sample_out': model_type == 'clf' and sample_out,
		}

	return suggestor

def get_suggest_model(base, sm_name, model_name, asset_name, dm):
	"""
	Dispatcher
	"""
	return {
		'anp': get_suggest_model_anp(base, model_name, asset_name, dm)
	}.get(sm_name)

def get_suggest_model_anp(base, model_name, asset_name, dm):
	"""
	suggest model hyperparameters from an optuna trial object
	or return fixed default hyperparameters
	"""
	def suggestor(trial):
		params = {}

		MIN_SIZE = 4
		ft_size = trial.suggest_int('ft_size', MIN_SIZE, 3*MIN_SIZE, step=MIN_SIZE) #3
		ft_global_dropout = trial.suggest_float('ft_global_dropout', .1, .5, step=.2) #3
		ft_init = trial.suggest_categorical('ft_init', \
			('kaiming_uniform', 'xavier_uniform', 'xavier_normal')) #3
		# ft_pad_mode = trial.suggest_categorical('ft_pad_mode', ('full', 'same'))
		params['ft_params'] = {
			'size': ft_size,
			'global_dropout': ft_global_dropout,
			'output_dropout': ft_global_dropout,
			'block_init': ft_init,
			'out_init': ft_init,
		}

		de_size_pwr = trial.suggest_int('de_size_pwr', 0, 2) #3
		de_size = (2**de_size_pwr) * 64
		de_depth = trial.suggest_int('de_depth', 2, 3) #2
		de_global_dropout = trial.suggest_float('de_global_dropout', .1, .5, step=.2) #3
		de_init = trial.suggest_categorical('de_init', \
			('kaiming_uniform', 'xavier_uniform', 'xavier_normal')) #3
		params['decoder_params'] = {
			'de_params': {
				'out_shapes': [de_size] * de_depth,
				'input_dropout': de_global_dropout,
				'global_dropout': de_global_dropout,
				'output_dropout': de_global_dropout,
				'init': de_init
			}
		}

		if (model_name.endswith('np')):
			params['sample_latent_post'] = trial.suggest_categorical(\
				'sample_latent_post', (False, True)) #2

		if (model_name in ('cnp', 'np')):
			det_class_agg = trial.suggest_categorical('det_class_agg', (False, True)) #2
			det_xa_num_heads_pwr = trial.suggest_int('det_xa_num_heads_pwr', 0, 2) #3
			det_rt_num_heads_pwr = trial.suggest_int('det_rt_num_heads_pwr', 0, 2) #3
			params['det_encoder_params'] = {
				'class_agg': det_class_agg,
				'xa_params': {
					'num_heads': ft_size // (2**det_xa_num_heads_pwr)
				},
				'rt_params': {
					'num_heads': ft_size // (2**det_rt_num_heads_pwr)
				}
			}
		if (model_name in ('lnp', 'np')):
			lat_class_agg = trial.suggest_categorical('lat_class_agg', (False, True)) #2
			# lat_min_std = trial.suggest_float('lat_min_std', .01, .05, step=.01)
			lat_rt_num_heads_pwr = trial.suggest_int('lat_rt_num_heads_pwr', 0, 2) #3
			params['lat_encoder_params'] = {
				'class_agg': lat_class_agg,
				# 'min_std': lat_min_std,
				'rt_params': {
					'num_heads': ft_size // (2**lat_rt_num_heads_pwr)
				}
			}

		return params

	return suggestor

# {
# 	"in_name": "in15d",
# 	"in_params": {
# 		"momentum": 0.1,
# 		"affine": false,
# 		"track_running_stats": false
# 	},
# 	"in_split": true,
# 	"fn_name": "in2d",
# 	"fn_params": {
# 		"momentum": 0.1,
# 		"affine": false,
# 		"track_running_stats": false
# 	},
# 	"fn_split": true,
# 	"ft_name": "stcn",
# 	"use_lvar": false,
# 	"label_size": 1,
# 	"out_size": 2,
# 	"num_classes": 2,
# 	"ft_params": {
# 		"size": 8,
# 		"depth": 3,
# 		"kernel_sizes": 17,
# 		"input_dropout": 0,
# 		"global_dropout": 0.5,
# 		"output_dropout": 0.5,
# 		"dropout_type": "2d",
# 		"dilation_factor": 2,
# 		"global_dilation": true,
# 		"block_act": "relu",
# 		"out_act": "relu",
# 		"block_init": "kaiming_uniform",
# 		"out_init": "kaiming_uniform",
# 		"pad_mode": "full"
# 	},
# 	"decoder_params": {
# 		"de_name": "ffn",
# 		"de_params": {
# 			"out_shapes": [
# 				64,
# 				64,
# 				64
# 			],
# 			"input_dropout": 0.2,
# 			"global_dropout": 0.2,
# 			"output_dropout": 0.2,
# 			"flatten": true,
# 			"act": "relu",
# 			"act_output": false,
# 			"init": "kaiming_uniform"
# 		},
# 		"dist_type": "beta",
# 		"min_std": 0.01,
# 		"use_lvar": false
# 	},
# 	"sample_latent_post": false,
# 	"sample_latent_prior": false,
# 	"det_encoder_params": {
# 		"class_agg": false,
# 		"xa_name": "mha",
# 		"xa_params": {
# 			"num_heads": 8,
# 			"dropout": 0,
# 			"depth": 1
# 		},
# 		"rt_name": "mha",
# 		"rt_params": {
# 			"num_heads": 8,
# 			"dropout": 0,
# 			"depth": 1
# 		}
# 	},
# 	"lat_encoder_params": {
# 		"latent_size": null,
# 		"cat_before_rt": false,
# 		"class_agg": false,
# 		"rt_name": "mha",
# 		"rt_params": {
# 			"num_heads": 8,
# 			"dropout": 0,
# 			"depth": 1
# 		},
# 		"dist_type": "normal",
# 		"min_std": 0.01,
# 		"use_lvar": false
# 	},
# 	"use_det_path": true,
# 	"use_lat_path": true
# }

