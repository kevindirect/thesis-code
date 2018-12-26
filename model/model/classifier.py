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

from common_util import MODEL_DIR, makedir_if_not_exists, in_debug_mode, remove_keys, dict_combine, dump_json, str_now, one_minus
from model.common import MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO, OPT_TRANSLATOR
from model.model.super_model import Model
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
		optimizer = OPT_TRANSLATOR.get(params['opt']['name'])
		return optimizer(lr=params['opt']['lr'])

	def make_const_data_objective(self, features, labels, exp_logdir, exp_meta=None, clf_type='binary', metaloss_type='val_acc', metaloss_mode='max',
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
			metaloss_type (str): the loss to return after the objective function is run
			metaloss_mode ('min'|'max'): whether the metaloss should be minimized or maximized
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
		exp_meta['data'] = {'size': labels.size, 'lab_dist': labels.value_counts(normalize=False).to_dict()}
		makedir_if_not_exists(exp_logdir)
		dump_json(exp_meta, 'exp.json', dir_path=exp_logdir)

		if (clf_type=='categorical' and labels.unique().size > 2):
			labels = pd.get_dummies(labels, drop_first=False) # If the labels are not binary (more than two value types), one hot encode them		
		feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, labels, test_ratio=test_ratio, shuffle=shuffle)

		def objective(params):
			"""
			Standard classifier objective function to minimize.
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

			if (in_debug_mode()):
				val_loss, val_acc = results['history']['val_loss'], results['history']['val_acc']
				logging.debug('val_loss mean, min, max, last: {mean}, {min}, {max}, {last}'
					.format(mean=np.mean(val_loss), min=np.min(val_loss), max=np.max(val_loss), last=val_loss[-1]))
				logging.debug('val_acc mean, min, max, last: {mean}, {min}, {max}, {last}'
					.format(mean=np.mean(val_acc), min=np.min(val_acc), max=np.max(val_acc), last=val_acc[-1]))

			metaloss = results['history'][metaloss_type][-1]	# Different from the loss used to fit models
			if (metaloss_mode == 'max'):	# Converts a score that should be maximized into a loss to minimize
				metaloss = one_minus(metaloss)

			# TODO - free GPU memory

			return {'loss': metaloss, 'status': STATUS_OK, 'params': params}

			# except:
			# 	return {'loss': ERROR_CODE, 'status': STATUS_OK, 'params': params}

		return objective

	def make_var_data_objective(self, raw_features, raw_labels, features_fn, labels_fn, exp_logdir, clf_type='binary', metaloss_type='val_acc',
								metaloss_mode='max', retain_holdout=True, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, shuffle=False):
		"""
		Return an objective function that hyperopt can use that can search over features and labels along with the hyperparameters.
		"""
		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			features, labels = features_fn(raw_features, params), labels_fn(raw_labels, params)
			return self.make_const_data_objective(features, labels, exp_logdir=exp_logdir, clf_type=clf_type, metaloss_type=metaloss_type,
													metaloss_mode=metaloss_mode, retain_holdout=retain_holdout, test_ratio=test_ratio, val_ratio=val_ratio,
													shuffle=shuffle)

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