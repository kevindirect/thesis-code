"""
Kevin Patel
"""
import sys
import os
from os import sep
import logging

import numpy as np
import pandas as pd
from dask import delayed

#from hyperopt import fmin, tpe, Trials
#from hyperopt.mongoexp import MongoTrials

from common_util import MODEL_DIR, JSON_SFX_LEN, makedir_if_not_exists, in_debug_mode, is_valid, get_class_name, str_to_list, get_cmd_args, load_json, benchmark
from model.common import FR_DIR, XG_DIR, DATASET_DIR, HOPT_WORKER_BIN, default_model, default_backend, default_dataset, default_trials_count
from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier

MODELS = {
	'rf': RandomForestClassifier
}

def fr():
	cmd_arg_list = ['models=', 'assets=', 'win_size=', 'class_weights=', 'target=', 'trials_count=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	models = str_to_list(cmd_input['models=']) if (is_valid(cmd_input['models='])) else 'rf'
	assets = str_to_list(cmd_input['assets=']) if (is_valid(cmd_input['assets='])) else ['sp_500', 'russell_2000', 'nasdaq_100', 'dow_jones']
	win_size = int(cmd_input['win_size=']) if (is_valid(cmd_input['win_size='])) else 1
	trials_count = int(cmd_input['trials_count=']) if (is_valid(cmd_input['trials_count='])) else 100

	for asset in assets:
		fd = get_xg_feature_dfs(asset)
		ld, td = get_xg_label_target_dfs(asset)
		for freq in ['h', 'd']:
			for src in ['pba', 'vol', 'buzz', 'nonbuzz']:
				# get all feature dfs for asset, freq, src combo
				# for each df, preproc and queue test
				makedir_if_not_exists(sep.join([FR_DIR, asset, freq, src]))
				# Concat dfs
				# preproc
				# get common rows
				# convert to numpy
				# K fold rolling CV
				# save results
	pass



if __name__ == '__main__':
	with benchmark('time to finish') as b:
		fr(sys.argv[1:])
