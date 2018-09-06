# Kevin Patel

import sys
import os
from os import sep
from functools import partial, reduce
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute

from common_util import RECON_DIR, DT_CAL_DAILY_FREQ, makedir_if_not_exists, set_loglevel, get_cmd_args, dump_df, load_json, outer_join, list_get_dict, benchmark
from recon.common import DATASET_DIR, REPORT_DIR, default_corr_dataset
from recon.dataset_util import prep_set
from recon.feat_util import gen_split_feats
from recon.label_util import shift_label, apply_label_mask, eod_fct, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct


def corr(argv):
	set_loglevel()
	cmd_arg_list = ['asset=', 'dataset=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='corr')
	chosen_asset = cmd_input['asset='] if (cmd_input['asset='] is not None) else None
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_corr_dataset
	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	dataset = prep_set(dataset_dict)
	results = []

	for corr_method in ['pearson', 'spearman', 'kendall']:

		for lpath in dataset['labels']['paths']:
			asset_name, base_label_name = lpath[0], lpath[-1]
			corr_matrix_name = '_'.join([corr_method, base_label_name])
			if (chosen_asset is not None and asset_name != chosen_asset):
				logging.info('skipping ' +asset_name +': ' +corr_matrix_name)
				continue
			else:
				logging.info(asset_name +': ' +corr_matrix_name)

			ldf = list_get_dict(dataset['labels']['dfs'], lpath)
			gldf = delayed(lambda d: d.groupby(pd.Grouper(freq=DT_CAL_DAILY_FREQ)).last())(ldf)

			eod = delayed(eod_fct)(gldf).add_suffix('_eod')
			fbeod = delayed(apply_label_mask)(gldf, default_fct).add_suffix('_fbeod')
			fb = delayed(apply_label_mask)(gldf, fastbreak_fct).add_suffix('_fb')
			conf = delayed(apply_label_mask)(gldf, confidence_fct).add_suffix('_conf')
			fbconf = delayed(apply_label_mask)(gldf, fastbreak_confidence_fct).add_suffix('_fbconf')
			bool_labels = [eod, fbeod, fb, conf, fbconf]

			vel = delayed(apply_label_mask)(gldf, partial(fastbreak_fct, velocity=True)).add_suffix('_vel')
			mag = delayed(apply_label_mask)(gldf, partial(confidence_fct, magnitude=True)).add_suffix('_mag')
			mom = delayed(apply_label_mask)(gldf, partial(fastbreak_confidence_fct, momentum=True)).add_suffix('_mom')
			int_labels = [vel, mag, mom]

			dest_dir = sep.join([REPORT_DIR, asset_name]) +sep
			makedir_if_not_exists(dest_dir)
			corr_matrix = delayed(pd.DataFrame)(columns=['feat_df_desc', 'feat_col_name'])

			for fpath in filter(lambda fpath: fpath[0]==asset_name, dataset['features']['paths']):
				feat_df_desc = fpath[-1]
				logging.debug(feat_df_desc)
				feats = list_get_dict(dataset['features']['dfs'], fpath)
				datadf = delayed(reduce)(outer_join, [feats, *bool_labels, *int_labels])
				sub_matrix = delayed(get_corr)(datadf, feat_df_desc, feats.columns, corr_method)
				corr_matrix = delayed(pd.concat)([corr_matrix, sub_matrix], axis=0, join='outer', ignore_index=True, sort=False)

			result = delayed(dump_df)(corr_matrix, corr_matrix_name, dir_path=dest_dir)
			results.append(result)

	logging.info('executing dask graph...')
	compute(*results)
	logging.info('done')


def get_corr(data, vert_desc, vert_cols, corr_method='pearson', label_shift_periods=-1):
	corr = pd.DataFrame()
	hor_cols = [col_name for col_name in data.columns if (col_name not in vert_cols)]

	for col_name in vert_cols:
		row = {}
		row['feat_df_desc'] = vert_desc
		row['feat_col_name'] = col_name
		row.update({hor_col: data[col_name].corr(shift_label(data[hor_col], shift_periods=label_shift_periods), method=corr_method) for hor_col in hor_cols})
		corr = corr.append(row, ignore_index=True)

	return corr


# def corr(argv):
# 	set_loglevel()
# 	cmd_arg_list = ['dataset=']
# 	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='corr')
# 	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_corr_dataset
# 	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
# 	dataset = prep_set(dataset_dict)
# 	results = []

# 	for lpath in dataset['labels']['paths']:
# 		asset_name, base_label_name = lpath[0], lpath[-1]
# 		logging.info(asset_name +' ' +base_label_name)
# 		ldf = list_get_dict(dataset['labels']['dfs'], lpath)
# 		gldf = delayed(lambda d: d.groupby(pd.Grouper(freq=DT_CAL_DAILY_FREQ)).last())(ldf)

# 		eod = delayed(eod_fct)(gldf).add_suffix('_eod')
# 		fbeod = delayed(apply_label_mask)(gldf, default_fct).add_suffix('_fbeod')
# 		fb = delayed(apply_label_mask)(gldf, fastbreak_fct).add_suffix('_fb')
# 		conf = delayed(apply_label_mask)(gldf, confidence_fct).add_suffix('_conf')
# 		fbconf = delayed(apply_label_mask)(gldf, fastbreak_confidence_fct).add_suffix('_fbconf')
# 		blabels = [eod, fbeod, fb, conf, fbconf]

# 		vel = delayed(apply_label_mask)(gldf, partial(fastbreak_fct, velocity=True)).add_suffix('_vel')
# 		mag = delayed(apply_label_mask)(gldf, partial(confidence_fct, magnitude=True)).add_suffix('_mag')
# 		mom = delayed(apply_label_mask)(gldf, partial(fastbreak_confidence_fct, momentum=True)).add_suffix('_mom')
# 		ilabels = [vel, mag, mom]

# 		for fpath in filter(lambda fpath: fpath[0]==asset_name, dataset['features']['paths']):
# 			logging.debug(fpath)
# 			feat_id = fpath[-1]
# 			feats = list_get_dict(dataset['features']['dfs'], fpath)
# 			datadf = delayed(reduce)(outer_join, [feats, *blabels, *ilabels])

# 			for corr_method in ['pearson', 'spearman', 'kendall']:
# 				dest_dir = sep.join([REPORT_DIR +corr_method, asset_name, base_label_name]) +sep
# 				makedir_if_not_exists(dest_dir)

# 				corr_mat = delayed(corr_matrix)(datadf, feats.columns, corr_method)
# 				result = delayed(dump_df)(corr_mat, feat_id, dir_path=dest_dir)
# 				results.append(result)

# 	logging.info('executing dask graph...')
# 	compute(*results)
# 	logging.info('done')


# def corr_matrix(data, vert_cols, corr_method='pearson', label_shift_periods=-1):
# 	corr = pd.DataFrame()
# 	hor_cols = [col_name for col_name in data.columns if (col_name not in vert_cols)]

# 	for col_name in vert_cols:
# 		row = {}
# 		row['index'] = col_name
# 		row.update({hor_col: data[col_name].corr(shift_label(data[hor_col], shift_periods=label_shift_periods), method=corr_method) for hor_col in hor_cols})
# 		corr = corr.append(row, ignore_index=True)

# 	return corr.set_index('index')


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		corr(sys.argv[1:])
