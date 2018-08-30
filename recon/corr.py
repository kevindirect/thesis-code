# Kevin Patel

import sys
import os
from itertools import product
import logging

import numpy as np
import pandas as pd
from dask import delayed

from common_util import RECON_DIR, get_cmd_args, load_json, list_get_dict, benchmark
from recon.common import DATASET_DIR, default_corr_dataset
from recon.dataset_util import prep_set
from recon.feat_util import gen_split_feats
from recon.label_util import gen_label_dfs, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct


def corr(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	cmd_arg_list = ['dataset=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='corr')
	dataset_name = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_corr_dataset

	dataset_dict = load_json(dataset_name, dir_path=DATASET_DIR)
	dataset = prep_set(dataset_dict)

	for lpath, fpath in product(dataset['labels']['paths'], dataset['features']['paths']):
		if (fpath[0] == lpath[0]): # Filter for combos where data source (asset) of label and feature df are the same
			print(fpath)
			print(lpath)
			# fdf = list_get_dict(dataset['features']['dfs'], fpath)
			ldf = list_get_dict(dataset['labels']['dfs'], lpath).compute()

			# create label columns from labdf (apply mask)
			labdf = delayed(apply_label_mask)(ldf, default_fct)

			# shift new label df
			

			# run correlation between feat df and label df

			# dump correlation matrix with filename identical to outer loop, in asset name folder




def corr_mat(df=None, feat_col_name=None, lab_col_name=None, **kwargs):
	cm = df.corr(**kwargs).loc[feat_col_name, lab_col_name]
	return cm


def corr_matrix(data, features, labels):
	corr = pd.DataFrame()
	for feat in features:
		row = {}
		row['feature'] = feat
		row.update({label: data[feat].corr(data[label]) for label in labels})
		corr = corr.append(row, ignore_index=True)
	return corr


# def corr_mat_1(df=None, feat_col_name=None, lab_col_name=None, **kwargs):
# 	for lab in lab_col_name:
# 		df[feat_col_name].corr()
# 		feat_col_name=None, lab_col_name=None,



if __name__ == '__main__':
	with benchmark('ttf ') as b:
		corr(sys.argv[1:])
