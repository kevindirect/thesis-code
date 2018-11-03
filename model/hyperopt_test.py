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


def hyperopt_test(argv):
	cmd_arg_list = ['dataset=', 'filterset=', 'idxfilters=', 'assets=', 'target_idx=', 'visualize']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='hyperopt_test')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	filterset_name = cmd_input['filterset='] if (cmd_input['filterset='] is not None) else '_'.join(['default', dataset_name])
	filter_idxs = str_to_list(cmd_input['idxfilters=']) if (cmd_input['idxfilters='] is not None) else default_opt_filter
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None
	target_idx = str_to_list(cmd_input['target_idx='], cast_to=int) if (cmd_input['target_idx='] is not None) else default_target_idx
	run_compute = True if (cmd_input['visualize'] is None) else False

	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	filter_dict = load_json(filterset_name, dir_path=FILTERSET_DIR)

	filterset = []
	for filter_idx in filter_idxs:
		selected = [flt for flt in filter_dict[filter_idx] if (flt not in filterset)]
		filterset.extend(selected)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map={'features': filterset})

	logging.info('assets: ' +str('all' if (assets==None) else ', '.join(assets)))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']['paths']), dataset_name[:-JSON_SFX_LEN]))
	logging.info('filter: {} [{}]'.format(filterset_name[:-JSON_SFX_LEN], str(', '.join(filter_idxs))))
	logging.debug('filterset: ' +str(filterset))
	logging.debug('fpaths: ' +str(dataset['features']['paths']))
	logging.debug('lpaths: ' +str(dataset['labels']['paths']))
	logging.debug('rmpaths: ' +str(dataset['row_masks']['paths']))

	labs_filter = [ # EOD, FBEOD, FB
	{
		"exact": [],
		"startswith": ["pba_"],
		"endswith": [],
		"regex": [],
		"exclude": None
	},
	{
		"exact": [],
		"startswith": [],
		"endswith": ["_eod(0%)", "_eod(1%)", "_eod(2%)", "_fb", "_fbeod"],
		"regex": [],
		"exclude": None
	}]

	dataset_grid = {
		'feat_idx': [0, 1, 2, 3, 4],
		'label_idx': target_idx
	}

	dataset_space = {
		'feat_idx': hp.choice('feat_idx', [0, 1, 2, 3, 4]),
		'label_idx': hp.choice('label_idx', target_idx)
	}

	if (run_compute):
		logging.info('executing...')
		for paths, dfs in gen_group(dataset):
			results = {}
			fpaths, lpaths, rpaths = paths
			features, labels, row_masks = dfs
			asset = fpaths[0]
			logging.info('fpaths: {}'.format(str(fpaths)))
			logging.info('lpaths: {}'.format(str(lpaths)))
			logging.info('rpaths: {}'.format(str(rpaths)))

			masked_labels = prepare_masked_labels(labels, ['bool'], labs_filter)

			for feat_idx, label_idx in product(*dataset_grid.values()):
				final_feature = prepare_transpose_data(features.iloc[:, [feat_idx]], row_masks).dropna(axis=0, how='all')
				shifted_label = delayed(shift_label)(masked_labels.iloc[:, label_idx]).dropna()
				pos_label, neg_label = delayed(pd_binary_clip, nout=2)(shifted_label)
				final_common = delayed(pd_common_index_rows)(final_feature, pos_label, neg_label)
				f, lpos, lneg = final_common.compute()

				logging.info('pos dir model experiment')
				run_trials(ThreeLayerBinaryFFN, f, lpos)

				logging.info('neg dir model experiment')
				run_trials(ThreeLayerBinaryFFN, f, lneg)

			# mod = OneLayerLSTM(dataset_space)
			# obj = mod.make_var_data_objective(features, labels

def run_trials(model_exp, features, label):
	mod = model_exp()
	obj = mod.make_const_data_objective(features, label)
	trials = Trials()
	best = fmin(obj, mod.get_space(), algo=tpe.suggest, max_evals=50, trials=trials)
	best_params = mod.params_idx_to_name(best)
	bad = mod.get_bad_trials()

	print('best idx: {}'.format(best))
	print('best params: {}'.format(best_params))
	if (bad > 0):
		print('bad trials: {}'.format(bad))

	return best_params


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		hyperopt_test(sys.argv[1:])
