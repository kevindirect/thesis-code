# Kevin Patel

import sys
import os
from os import sep
from os.path import splitext
from functools import partial, reduce
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute, visualize

from common_util import RECON_DIR, JSON_SFX_LEN, get_cmd_args, get_variants, dump_df, load_json, outer_join, list_get_dict, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, default_dataset, default_filterset, default_filter
from recon.dataset_util import prep_dataset, prep_labels


def cnn(argv):
	cmd_arg_list = ['dataset=', 'filterset=', 'idxfilters=', 'assets=', 'visualize']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='cnn')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	filterset_name = cmd_input['filterset='] if (cmd_input['filterset='] is not None) else default_filterset
	filter_idxs =  list(map(str.strip, cmd_input['idxfilters='].split(','))) if (cmd_input['idxfilters='] is not None) else default_filter
	assets = list(map(str.strip, cmd_input['assets='].split(','))) if (cmd_input['assets='] is not None) else None
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

	if (run_compute):
		logging.info('executing...')
		cnn_test(dataset)
		logging.info('done')
	else:
		# logging.info('visualizing first dask graph...')
		# visualize(tests[0], filename='first_graph.svg')
		# logging.info('visualizing last dask graph...')
		# visualize(tests[-1], filename='last_graph.svg')
		# logging.info('done')


def cnn_test(dataset):

	for lpath in dataset['labels']['paths']:
		asset_name, base_label_name = lpath[0], lpath[-1]
		test_name = '_'.join([meta['pfx_fmt'].format(**variant), base_label_name])
		logging.info(asset_name +': ' +test_name)

		label_df = list_get_dict(dataset['labels']['dfs'], lpath)
		labels = prep_labels(label_df)

		for fpath in filter(lambda fpath: fpath[0]==asset_name, dataset['features']['paths']):
			feat_df_desc = fpath[-1]
			logging.debug(feat_df_desc)
			feats = list_get_dict(dataset['features']['dfs'], fpath)
			sub_matrix = delayed(gta_apply_type_fn)(feats, feat_df_desc, labels, partial(gta_test_fn, **variant))
			matrix = delayed(pd.concat)([matrix, sub_matrix], axis=0, join='outer', ignore_index=True, sort=False)


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		cnn(sys.argv[1:])
