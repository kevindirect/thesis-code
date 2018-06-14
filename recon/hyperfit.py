# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from dask_ml.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, f1_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, roc_curve, precision_recall_curve

from common_util import remove_dups_list, list_get_dict
from recon.common import dum
from recon.cv_util import PurgedKFold
from recon.model_util import MyPipeline


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0,None,1.], rnd_search_iter=0, n_jobs=-1, pct_embargo=0, **fit_params):
	"""
	Randomized Search with Purged K-Fold CV
	Lopez De Prado, Advances in Financial Machine Learning (p. 131-132)

	Args:

	Returns:
		
	"""
	if (set(lbl.values)=={0,1}):
		scoring = 'f1'
	else:
		scoring = 'neg_log_loss'

	#1) Hyperparameter search on training data
	inner_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
	if (rnd_search_iter == 0):
		gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
	else:
		gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
			iid=False, n_iter=rnd_search_iter)
	gs.fit(feat, lbl, **fit_params).best_estimator_

	#2) Fit validated model on the entirety of the data
	if (bagging[1]>0):
		gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
			max_features=float(bagging[2]), n_jobs=n_jobs)
		gs = gs.fit(feat, lbl, sample_weight=fit_params [gs.base_estimator.steps[-1][0]+'__sample_weight'])
		gs = Pipeline([('bag', gs)])

	return gs



def cv_hyper_fit(feat_mat, label_arr, pipe_clf, param_grid, cv_splitter, scoring='neg log loss'):
	"""
	Optimizes hyperparameters of the classifier according to the passed options.
	
	Args:
		feat_map (np.array)
		label_arr (np.array)
		pipe_clf
	"""

	gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=cv_splitter, n_jobs=-1, iid=False)

	return gs