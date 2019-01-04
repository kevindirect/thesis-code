"""
Kevin Patel
"""
import sys
import os
from os import sep
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK, STATUS_FAIL
from keras.optimizers import SGD, RMSprop, Adam, Nadam

from common_util import MODEL_DIR, makedir_if_not_exists, remove_keys, dict_combine, dump_json, str_now, one_minus
from model.common import MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO, KERAS_OPT_TRANSLATOR
from model.model_k.keras_model import Model
from recon.split_util import get_train_test_split


class Classifier(Model):
	"""Abstract Base Class of all classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'opt': hp.choice('opt', [
				{'name': 'RMSprop', 'lr': hp.choice('RMSprop_lr', [0.002, 0.001, 0.0005])},
				{'name': 'Adam', 'lr': hp.choice('Adam_lr', [0.002, 0.001, 0.0005])},
				{'name': 'Nadam', 'lr': hp.choice('Nadam_lr', [0.002])},
			])
		}
		super(Classifier, self).__init__({**default_space, **other_space})
		self.metrics = ['accuracy']

	def make_optimizer(self, params):
		"""
		Converts optimizer parameters into a function that can be passed to keras model compile.
		"""
		optimizer = KERAS_OPT_TRANSLATOR.get(params['opt']['name'])
		return optimizer(lr=params['opt']['lr'])

	def make_const_data_objective(self, features, labels, exp_logdir, exp_meta=None, clf_type='binary',
									meta_obj='val_acc', obj_agg='last', obj_mode='max', meta_var=None,
									retain_holdout=True, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, shuffle=False):
		"""
		Return an objective function that hyperopt can use for the given features and labels.
		Acts as a factory for an objective function of a model over params.
		Logs metadata files and trial subdirectories in exp_logdir.

		Args:
			features (pd.DataFrame): features df
			labels (pd.DataFrame): labels df
			exp_logdir (str): path to the logging directory of the objective function
			exp_meta (dict): any additional key-value metadata to log for the experiment (locals are always logged)
			clf_type ('binary'|'categorical'): the classifier type 
			meta_obj (str): the name of the loss to return after the objective function is run
			obj_agg ('last'|'mean'): the method to aggregate the objective function
			obj_mode ('min'|'max'): indicates whether the meta objective should be minimized or maximized
			meta_var (None | str): the name of the loss uncertainty variable, if unused uncertainty will be fixed to zero (point estimate model)
			retain_holdout (bool): if true the the test set is held out and validation set is taken from the training set,
									if false the test set is used as validation and there is no test set
			test_ratio (float): ratio to be removed for test
			val_ratio (float): ratio to be removed for validation (only used if retain_holdout set true)
			shuffle (bool): if true shuffles the data

		Returns:
			objective function to arg minimize
		"""
		exp_meta = exp_meta or {}
		exp_meta['params'] = remove_keys(dict(locals().items()), ['self', 'features', 'labels', 'exp_meta'])
		exp_meta['data'] = {'size': labels.size, 'lab_dist': labels.value_counts(normalize=True).to_dict()}
		makedir_if_not_exists(exp_logdir)
		dump_json(exp_meta, 'exp.json', dir_path=exp_logdir)

		if (clf_type=='categorical' and labels.unique().size > 2):
			labels = pd.get_dummies(labels, drop_first=False) # If the labels are not binary (more than two value types), one hot encode them		
		feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, labels, test_ratio=test_ratio, shuffle=shuffle)

		def objective(params):
			"""
			Standard classifier objective function to minimize.

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
			compiled = self.get_model(params, features.shape[1])
			trial_logdir = exp_logdir +str_now() +sep
			makedir_if_not_exists(trial_logdir)
			dump_json(params, 'params.json', dir_path=trial_logdir)

			if (retain_holdout):
				results = self.fit_model(params, trial_logdir, compiled, (feat_train, lab_train), val_data=None, val_split=val_ratio, shuffle=shuffle)
			else:
				results = self.fit_model(params, trial_logdir, compiled, (feat_train, lab_train), val_data=(feat_test, lab_test), val_split=val_ratio, shuffle=shuffle)

			metaloss = results[obj_agg][meta_obj]										# Different from the loss used to fit models
			metaloss_var = 0 if (meta_var is None) else results[obj_agg][meta_var]		# If zero the prediction is a point estimate
			if (obj_mode == 'max'):														# Converts a score that to maximize into a loss to minimize
				metaloss = one_minus(metaloss)

			# TODO - free GPU memory

			return {'status': STATUS_OK, 'loss': metaloss, 'loss_variance': metaloss_var, 'params': params}

			# except:
			# 	return {'status': STATUS_OK, 'loss': ERROR_CODE, 'loss_variance': ERROR_CODE, 'params': params}

		return objective

	def make_var_data_objective(self, raw_features, raw_labels, features_fn, labels_fn, exp_logdir, exp_meta=None, clf_type='binary',
									meta_obj='val_acc', obj_agg='last', obj_mode='max', meta_var=None,
									retain_holdout=True, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, shuffle=False):		
		"""
		Return an objective function that hyperopt can use that can search over features and labels along with the hyperparameters.
		"""
		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			features, labels = features_fn(raw_features, params), labels_fn(raw_labels, params)
			return self.make_const_data_objective(features, labels, exp_logdir=exp_logdir, exp_meta=exp_meta, clf_type=clf_type,
													meta_obj=meta_obj, obj_agg=obj_agg, obj_mode=obj_mode, meta_var=meta_var,
													retain_holdout=retain_holdout, test_ratio=test_ratio, val_ratio=val_ratio, shuffle=shuffle)

		return objective

	def make_ray_objective(self, objective):
		"""
		Return an objective function that can be used by ray based on a hyperopt objective.
		"""
		def ray_objective(params, reporter):
			"""
			Ray objective function requires the passing of the reporter object.
			Note in the Ray doc examples "params" is called "config".
			Also Ray is built on reward functions rather than loss functions.
			"""
			return reporter(reward=-objective(params)['loss'])

		return ray_objective