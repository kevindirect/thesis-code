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
	cmd_arg_list = ['dataset=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='corr')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_corr_dataset
	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	dataset = prep_set(dataset_dict)
	results = []

	for lpath in dataset['labels']['paths']:
		asset_name, base_label_name = lpath[0], lpath[-1]
		logging.info(asset_name +' ' +base_label_name)
		ldf = list_get_dict(dataset['labels']['dfs'], lpath)
		gldf = delayed(lambda d: d.groupby(pd.Grouper(freq=DT_CAL_DAILY_FREQ)).last())(ldf)

		eod = delayed(eod_fct)(gldf).add_suffix('_eod')
		fbeod = delayed(apply_label_mask)(gldf, default_fct).add_suffix('_fbeod')
		fb = delayed(apply_label_mask)(gldf, fastbreak_fct).add_suffix('_fb')
		conf = delayed(apply_label_mask)(gldf, confidence_fct).add_suffix('_conf')
		fbconf = delayed(apply_label_mask)(gldf, fastbreak_confidence_fct).add_suffix('_fbconf')
		blabels = [eod, fbeod, fb, conf, fbconf]

		vel = delayed(apply_label_mask)(gldf, partial(fastbreak_fct, velocity=True)).add_suffix('_fbv')
		mag = delayed(apply_label_mask)(gldf, partial(confidence_fct, magnitude=True)).add_suffix('_confl')
		mom = delayed(apply_label_mask)(gldf, partial(fastbreak_confidence_fct, momentum=True)).add_suffix('_mom')
		ilabels = [vel, mag, mom]

		for fpath in filter(lambda fpath: fpath[0]==asset_name, dataset['features']['paths']):
			logging.debug(fpath)
			feat_id = fpath[-1]
			feats = list_get_dict(dataset['features']['dfs'], fpath)
			datadf = delayed(reduce)(outer_join, [feats, *blabels, *ilabels])

			for corr_method in ['pearson', 'spearman', 'kendall']:
				dest_dir = sep.join([REPORT_DIR +corr_method, asset_name, base_label_name]) +sep
				makedir_if_not_exists(dest_dir)

				corr_mat = delayed(corr_matrix)(datadf, feats.columns, corr_method)
				print(corr_mat.compute())
	# 			result = delayed(dump_df)(corr_mat, feat_id, dir_path=dest_dir, data_format='csv')
	# 			results.append(result)

	# for res in results:
	# 	size = res.compute()
	# 	logging.info(size)


# def eod_label(df1, gb_freq=DT_CAL_DAILY_FREQ):
# 	eodf = delayed(eod_fct)(df1)
# 	gbdf = delayed(lambda d: d.groupby(pd.Grouper(freq=gb_freq)).last())(eodf)
# 	return gbdf

# def corr_dfs(df1, df2):
# 	correls = df1.corrwith(df2, axis=0, drop=True)
# 	return correls

def corr_matrix(data, vert_cols, corr_method='pearson', label_shift_periods=-1):
	corr = pd.DataFrame()
	hor_cols = [col_name for col_name in data.columns if (col_name not in vert_cols)]

	for col_name in vert_cols:
		row = {}
		row['index'] = col_name
		row.update({hor_col: data[col_name].corr(shift_label(data[hor_col], shift_periods=label_shift_periods), method=corr_method) for hor_col in hor_cols})
		corr = corr.append(row, ignore_index=True)

	return corr.set_index('index')


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		corr(sys.argv[1:])
