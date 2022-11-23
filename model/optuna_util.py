import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import optuna
from optuna.pruners import PercentilePruner, HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback

from common_util import MODEL_DIR, is_type, is_valid, isnt
from model.common import ASSETS

# Samplers
def get_sampler(sm_name, model_name, sampler_type='tpe'):
	if (sampler_type == 'grid'):
		sampler = optuna.samplers.GridSampler(get_search_grid(sm_name, model_name))
	elif (sampler_type == 'tpe'):
		sampler = optuna.samplers.TPESampler()
	return sampler

def get_search_grid(sm_name, model_name):
	return {
		'anp': get_search_grid_anp(model_name)
	}.get(sm_name)

def get_search_grid_anp(model_name):
	if (model_name == 'np'):
		raise NotImplementedError()
		grid = {
			
		}
	return grid

# Suggestors
def get_model_suggestor(sm_name, model_name):
	return {
		'anp': get_model_suggestor_anp(model_name)
	}.get(sm_name)

def get_model_suggestor_anp(model_name):
	def suggestor(trial):
		embed_size = 128 * (2**trial.suggest_int('embed_size', 0, 2))
		use_lvar = trial.suggest_categorical('luse_lvar', (True, False))
		init = trial.suggest_categorical('init', \
			('xavier_uniform', 'kaiming_uniform', 'xavier_normal', 'kaiming_normal'))
		ft_params_ob_params_depth = trial.suggest_int('ft_params_ob_params_depth ', 1, 2)
		decoder_params_de_params_depth = trial.suggest_int('decoder_params_de_params_depth ', 2, 5)

		rt_name = trial.suggest_categorical('rt_name', (None, 'mha', 'ffn'))
		if (rt_name == 'mha'):
			rt_params = {
				"num_heads": 1,
				"dropout": 0.0,
				"depth": 1
			}
		elif (rt_name == 'ffn'):
			rt_params = {
				"init": init,
				"out_shapes": [embed_size] * trial.suggest_int('rt_params_ffn_depth ', 1, 3),
				"flatten": False,
				"act": "relu",
				"act_output": trial.suggest_categorical('rt_params_ffn_act_output', (True, False))
			}
		else:
			rt_params = {}

		params = {
			"opt": {
				"kwargs": {
					"lr": trial.suggest_float('opt_kwargs_lr', 2e-5, 2e-4, log=True)
				}
			},
			"ft_params": {
				"block_init": init,
				"out_init": init,
				"size": trial.suggest_int('ft_params_size', 2, 8, 2),
				"depth": trial.suggest_int('ft_params_depth', 3, 5),
				"kernel_sizes": trial.suggest_int('ft_params_kernel_sizes', 5, 30, 5),
				"collapse_out": trial.suggest_categorical('ft_params_collapse_out', (True, False)),
				"ob_params": {
					"init": init,
					"out_shapes": [embed_size] * ft_params_ob_params_depth,
					"act_output": trial.suggest_categorical('ft_params_ob_params_act_output', (True, False))
				}
			},
			"decoder_params": {
				"de_params": {
					"init": init,
					"out_shapes": [embed_size] * decoder_params_de_params_depth,
					"act_output": trial.suggest_categorical('decoder_params_de_params_act_output', (True, False)),
				},
				"act": trial.suggest_categorical('decoder_params_act', (None, "tanhshrink")),
				"min_std": trial.suggest_float('decoder_params_min_std', 1e-9, 1e-1, log=True),
				"use_lvar": use_lvar
			},
			"det_encoder": {
				"rt_name": rt_name,
				"rt_params": rt_params,
				"xa_params":{
					"num_heads": 2**trial.suggest_int('det_encoder_xa_params_num_heads_pow', 0, 4),
				}
			},
			"lat_encoder": {
				"latent_size": embed_size,
				"rt_name": rt_name,
				"rt_params": rt_params,
				"min_std": trial.suggest_float('lat_encoder_params_min_std', 1e-9, 1e-1, log=True),
				"use_lvar": use_lvar
			}
		}

		return params

	return suggestor
