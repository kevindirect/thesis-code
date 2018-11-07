"""
Kevin Patel
"""

import sys
import os
from os import sep
from os.path import splitext
from itertools import product
from functools import partial, reduce
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute, visualize
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from common_util import MODEL_DIR, RECON_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, str_to_list, get_cmd_args, in_debug_mode, pd_common_index_rows, load_json, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, default_dataset, default_opt_filter, default_target_idx
from model.model_util import prepare_transpose_data, prepare_masked_labels
from model.models.ThreeLayerBinaryFFN import ThreeLayerBinaryFFN
from model.models.OneLayerBinaryLSTM import OneLayerBinaryLSTM
from recon.dataset_util import prep_dataset, prep_labels, gen_group
from recon.split_util import get_train_test_split, pd_binary_clip
from recon.label_util import shift_label


def run_exp(argv):
	cmd_arg_list = ['exp_list=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='run_exp')
	experiments = cmd_input['exp_list='] if (cmd_input['exp_list='] is not None) else default_exp

	for experiment in exp:
		experiment = exp()
		experiment.run_trials()
		experiment.



def run_trials(model_exp, features, label):
	exp = model_exp()
	trials = Trials()
	obj = exp.make_const_data_objective(features, label)
	best = fmin(obj, exp.get_space(), algo=tpe.suggest, max_evals=50, trials=trials)
	best_params = exp.params_idx_to_name(best)
	bad = exp.get_bad_trials()

	print('best idx: {}'.format(best))
	print('best params: {}'.format(best_params))
	if (bad > 0):
		print('bad trials: {}'.format(bad))

	return best_params


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		run_exp(sys.argv[1:])
