# Kevin Patel

import sys
import os
import getopt
from operator import itemgetter
import logging

import numpy as np
import pandas as pd

from common_util import RECON_DIR, get_cmd_args, load_json, dump_df, inner_join, remove_dups_list, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import default_corr_dataset
from recon.feat_util import gen_split_feats
from recon.label_util import gen_label_dfs, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct


def corr(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	cmd_arg_list = ['dataset=', 'all']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='corr')
	dataset = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_corr_dataset
	# runt_all = True if (cmd_input['all'] is not None) else False

	prep_set(dataset)

	


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
