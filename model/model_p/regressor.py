"""
Kevin Patel
"""
import sys
import os
from os import sep
from functools import partial
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from hyperopt import hp, STATUS_OK, STATUS_FAIL

from common_util import MODEL_DIR, isnt, makedir_if_not_exists, remove_keys, dump_json, str_now, one_minus, is_type, midx_split, pd_midx_to_arr
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, VAL_RATIO, TEST_RATIO, PYTORCH_LOSS_TRANSLATOR, PYTORCH_OPT_TRANSLATOR
from model.model_p.pytorch_model import Model
from recon.split_util import index_three_split


class Regressor(Model):
	"""
	Abstract Base Class of all regressors.

	Parameters:
		loss (string): string representing pytorch loss function to use
		opt (string): string representing pytorch optimizer to use
		opt.lr (float > 0): optimizer learning rate
	"""
	def __init__(self, other_space={}):
		default_space = {
			'loss': hp.choice('loss', ['mae']),
			'opt': hp.choice('opt', [
			# 	{'name': 'RMSprop', 'lr': hp.choice('RMSprop_lr', [0.002, 0.001, 0.0005])},
				{'name': 'Adam', 'lr': hp.uniform('Adam_lr', 0.0005, 0.0025)}
			])
		}
		super(Regressor, self).__init__({**default_space, **other_space})
		self.metrics_fns = {
			'mae': mean_absolute_error,
			'mdae': median_absolute_error,
			'mse': mean_squared_error,
			'r2': r2_score
		}

	def make_loss_fn(self, params):
		"""
		Make pytorch loss function object based on passed params.
		"""
		loss_fn = PYTORCH_LOSS_TRANSLATOR.get(params['loss'])
		return loss_fn()

	def make_optimizer(self, params, model_params):
		"""
		Make pytorch optimizer object over model_params based on passed params.
		"""
		optimizer = PYTORCH_OPT_TRANSLATOR.get(params['opt']['name'])
		return optimizer(model_params, lr=params['opt']['lr'])

	def make_const_data_objective(self, features, targets, exp_logdir=None, exp_meta=None, reg_type=None,
									meta_obj='val_loss', obj_agg='last', obj_mode='min', meta_var=None,
									val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, shuffle=False):
		"""
		Return an objective function that hyperopt can use for the given features and targets.
		Acts as a factory for an objective function of a model over params.
		Logs metadata files and trial subdirectories in exp_logdir.

		Args:
			features (pd.DataFrame): features df
			targets (pd.DataFrame or pd.Series): targets df or series
			exp_logdir (str): path to the logging directory of the objective function (no logging if not supplied)
			exp_meta (dict): any additional key-value metadata to log for the experiment (locals are automatically logged)
			reg_type (): the regressor type
			meta_obj (str): the name of the loss to return after the objective function is run
			obj_agg ('last'|'mean'): the method to aggregate the objective function
			obj_mode ('min'|'max'): indicates whether the meta objective should be minimized or maximized
			meta_var (None | str): the name of the loss uncertainty variable, if unused uncertainty will be fixed to zero (point estimate model)
			val_ratio (float): ratio to be removed for validation
			test_ratio (float): ratio to be removed for test
			shuffle (bool): if true shuffles the data

		Returns:
			objective function to arg minimize
		"""
		exp_meta = exp_meta or {}
		exp_meta['params'] = remove_keys(dict(locals().items()), ['self', 'features', 'targets', 'exp_meta'])
		exp_meta['data'] = {'size': targets.size, 'lab_dist': targets.value_counts(normalize=True).to_dict()}
		if (exp_logdir is not None):
			makedir_if_not_exists(exp_logdir)
			dump_json(exp_meta, 'exp.json', dir_path=exp_logdir)

		# XXX - shuffle option (probably won't need)
		# XXX - create the dataset dataloaders here?
		# XXX - encapsulate this data prep into a function?
		train_ratio = 1-(val_ratio+test_ratio)
		f_train_idx, f_val_idx, f_test_idx = midx_split(features.index, train_ratio, val_ratio, test_ratio)
		l_train_idx, l_val_idx, l_test_idx = midx_split(targets.index, train_ratio, val_ratio, test_ratio)
		# t_train_idx, t_val_idx, t_test_idx = midx_split(t.index, train_ratio, val_ratio, test_ratio)

		f_train_pd, f_val_pd, f_test_pd = features.loc[f_train_idx], features.loc[f_val_idx], features.loc[f_test_idx]
		l_train_pd, l_val_pd, l_test_pd = targets.loc[l_train_idx], targets.loc[l_val_idx], targets.loc[l_test_idx]
		# t_train_pd, t_val_pd, t_test_pd = targets.loc[t_train_idx], targets.loc[t_val_idx], targets.loc[t_test_idx]

		if (is_type(features.index, pd.core.index.MultiIndex)):
			f_train_np, f_val_np, f_test_np = map(pd_midx_to_arr, [f_train_pd.stack(), f_val_pd.stack(), f_test_pd.stack()])
		else:
			f_train_np, f_val_np, f_test_np = f_train_pd.values, f_val_pd.values, f_test_pd.values
		l_train_np, l_val_np, l_test_np = l_train_pd.values, l_val_pd.values, l_test_pd.values
		# t_train_np, t_val_np, t_test_np = t_train_pd.values, t_val_pd.values, t_test_pd.values
		input_shape = tuple(f_train_np.shape[-2:]) if (len(f_train_np.shape) < 3) else (1, f_train_np.shape[-1])

		def objective(params):
			"""
			Standard regressor objective function to minimize.

			Args:
				params (dict): dictionary of objective function parameters

			Returns:
				dict containing the following items
						status: one of the keys from hyperopt.STATUS_STRINGS, will be fixed to STATUS_OK
						loss: the float valued loss (validation set loss), if status is STATUS_OK this has to be present
						loss_variance: the uncertainty in a stochastic objective function
						params: the params passed in (for convenience)
					optional:
						true_loss: generalization error (test set loss), not actually used in tuning
						true_loss_variance: uncertainty of the generalization error, not actually used in tuning
			"""
			# try:
			trial_logdir = str(exp_logdir +str_now() +sep) if (exp_logdir is not None) else None
			if (trial_logdir is not None):
				makedir_if_not_exists(trial_logdir)
				dump_json(params, 'params.json', dir_path=trial_logdir)

			dev = torch.device('cuda') if (torch.cuda.is_available()) else torch.device('cpu')
			mdl = self.get_model(params, input_shape).to(device=dev)
			res = self.fit_model(params, trial_logdir, mdl, dev, (f_train_np, l_train_np), val_data=(f_val_np, l_val_np))
			# dump_json(res, 'results.json', dir_path=trial_logdir)

			metaloss = res[obj_agg][meta_obj]										# Different from the loss used to fit models
			metaloss_var = 0 if (meta_var is None) else res[obj_agg][meta_var]		# If zero the prediction is a point estimate
			if (obj_mode == 'max'):													# Converts a score that to maximize into a loss to minimize
				metaloss = one_minus(metaloss)

			return {'status': STATUS_OK, 'loss': metaloss, 'loss_variance': metaloss_var, 'params': params}

			# except:
			# 	return {'status': STATUS_OK, 'loss': ERROR_CODE, 'loss_variance': ERROR_CODE, 'params': params}

		return objective

	def make_var_data_objective(self, raw_features, raw_targets, features_fn, targets_fn, exp_logdir, exp_meta=None, reg_type=None,
									meta_obj='val_acc', obj_agg='last', obj_mode='max', meta_var=None,
									retain_holdout=True, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, shuffle=False):
		"""
		Return an objective function that hyperopt can use that can search over features and targets along with the hyperparameters.
		"""
		def objective(params):
			"""
			Standard regressor objective function to minimize.
			"""
			features, targets = features_fn(raw_features, params), targets_fn(raw_targets, params)
			return self.make_const_data_objective(features, targets, exp_logdir=exp_logdir, exp_meta=exp_meta, reg_type=reg_type,
													meta_obj=meta_obj, obj_agg=obj_agg, obj_mode=obj_mode, meta_var=meta_var,
													retain_holdout=retain_holdout, test_ratio=test_ratio, val_ratio=val_ratio, shuffle=shuffle)

		return objective
