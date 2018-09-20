# Kevin Patel

import sys
import os
from os import sep
from os.path import splitext
from functools import partial
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute, visualize

from common_util import RECON_DIR, get_cmd_args, makedir_if_not_exists, flatten2D, get_variants, dump_df, load_json, outer_join, list_get_dict, benchmark
from recon.common import DATASET_DIR, TEST_DIR, default_gta_test, default_gta_dataset
from recon.dataset_util import prep_dataset, prep_labels
from recon.gta_util import GTA_TYPE_TRANSLATOR, GTA_TEST_TRANSLATOR, report_path_dir


def generic_test_applicator(argv):
	cmd_arg_list = ['dataset=', 'test=', 'assets=', 'visualize']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='gta')
	test_name = cmd_input['test='] if (cmd_input['test='] is not None) else default_gta_test
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_gta_dataset
	assets = list(map(str.strip, cmd_input['assets='].split(','))) if (cmd_input['assets='] is not None) else None
	run_compute = True if (cmd_input['visualize'] is None) else False

	test_spec = load_json(test_name, dir_path=TEST_DIR)
	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	dataset = prep_dataset(dataset_dict, assets=assets)

	logging.info('assets: ' +str('all' if (assets==None) else ', '.join(assets)))
	logging.info('dataset name: ' +dataset_name)
	logging.info('test name: ' +test_name)

	if (isinstance(test_spec, dict)):
		tests = specify_test(test_spec, dataset, dataset_name=dataset_name)
	elif (isinstance(test_spec, list)):
		tests = []
		for subtest_name in test_spec:
			logging.info('subtest name: ' +subtest_name)
			subtest_spec = load_json(subtest_name, dir_path=TEST_DIR)
			tests.extend(specify_test(subtest_spec, dataset, dataset_name=dataset_name))

	if (run_compute):
		logging.info('executing dask graphs...')
		compute(*tests)
		logging.info('done')
	else:
		logging.info('visualizing first dask graph...')
		visualize(tests[0], filename='first_graph.svg')
		logging.info('visualizing last dask graph...')
		visualize(tests[-1], filename='last_graph.svg')
		logging.info('done')


def specify_test(test_dict, dataset, dataset_name):
	meta, fn, var = test_dict['meta'], test_dict['fn'], test_dict['var']
	gta_apply_type_fn, gta_test_fn = GTA_TYPE_TRANSLATOR[fn['type_fn']], GTA_TEST_TRANSLATOR[fn['test_fn']]
	variants = get_variants(var, meta['var_fmt'])
	tests = []
	logging.debug('parameter space: ' +str(variants))

	for variant in variants:
		for lpath in dataset['labels']['paths']:
			asset_name, base_label_name = lpath[0], lpath[-1]
			test_name = '_'.join([meta['pfx_fmt'].format(**variant), base_label_name])
			logging.info(asset_name +': ' +test_name)

			matrix = delayed(pd.DataFrame)(columns=['feat_df_desc', 'feat_col_name'])
			label_df = list_get_dict(dataset['labels']['dfs'], lpath)
			labels = prep_labels(label_df)

			for fpath in filter(lambda fpath: fpath[0]==asset_name, dataset['features']['paths']):
				feat_df_desc = fpath[-1]
				logging.debug(feat_df_desc)
				feats = list_get_dict(dataset['features']['dfs'], fpath)
				sub_matrix = delayed(gta_apply_type_fn)(feats, feat_df_desc, labels, partial(gta_test_fn, **variant))
				matrix = delayed(pd.concat)([matrix, sub_matrix], axis=0, join='outer', ignore_index=True, sort=False)

			dest_dir = report_path_dir(dataset_name, asset_name)
			makedir_if_not_exists(dest_dir)
			test = delayed(dump_df)(matrix, test_name, dir_path=dest_dir)
			tests.append(test)

	return tests

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		generic_test_applicator(sys.argv[1:])
