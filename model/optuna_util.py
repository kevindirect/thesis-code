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
		# size = 128 * (2**trial.suggest_int('size', 0, 2))
		# size = 256
		# init = trial.suggest_categorical('init', ('kaiming_uniform', 'kaiming_normal'))
		# init = 'kaiming_normal'
		# use_lvar = trial.suggest_categorical('use_lvar', (True, False))
		# use_lvar = True
		# rt_name = trial.suggest_categorical('rt_name', (None, 'sffn'))
		rt_name = 'sffn'
		if (rt_name == 'sffn'):
			rt = {
				# "init": init,
				# "size": size,
				"depth": trial.suggest_int('rt_depth', 2, 3),
				# "act_output": trial.suggest_categorical('rt_params_act_output', (True, False))
			}
		else:
			rt = {}

		ft = {
			# "block_init": init,
			# "out_init": init,
			"size": 2**trial.suggest_int('ft_params_size_pow', 2, 3),
			# "depth": 3,
			"kernel_sizes": trial.suggest_categorical('ft_params_kernel_sizes', (5, 15, 30)),
			# "collapse_out": trial.suggest_categorical('ft_params_collapse_out', (True, False)),
			# "ob_params": {
				# "init": init,
				# "size": size,
				# "depth": 2
			# }
		}
		decoder = {
			# "de_params": {
				# "init": init,
				# "size": size,
				# "depth": 4
			# },
			# "use_lvar": use_lvar,
			"min_std": 10**-trial.suggest_int('decoder_params_min_std_npow', 1, 9)
		}
		return {
			"sample_out": trial.suggest_categorical('sample_out', (True, False)),
			"ft_params": ft,
			"decoder_params": decoder,
			"det_encoder_params": {
				# "rt_name": rt_name,
				"rt_params": rt,
				# "xa_params": {
					# "num_heads": 2**trial.suggest_int('det_encoder_xa_params_num_heads_pow', 0, 3)
					# "num_heads": 2
				# }
			},
			"lat_encoder_params": {
				# "latent_size": size,
				# "rt_name": rt_name,
				"rt_params": rt,
				# "use_lvar": use_lvar,
				"min_std": 10**-trial.suggest_int('lat_encoder_params_min_std_npow', 1, 9)
			}
		}

	return suggestor
