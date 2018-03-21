# Kevin Patel

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_util import DATA_DIR, benchmark, inner_join
from data.data_api import DataAPI
from eda.common import dum


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def main(argv):

	search_terms = {
		'stage': 'raw',
		'basis': 'sp_500'
	}
	for rec, df in DataAPI.generate(search_terms):
		print(rec.name)
		labs = make_label_cols(df)
		joined = inner_join(df, labs)

		with benchmark('corr_mat', suppress=False) as b:
			print(rec.name)
			out = joined.pipe(corr_mat_1, list(df.columns), list(labs.columns), method='spearman')
			print(out.head())



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

def make_label_cols(df):
	label_cols = pd.DataFrame(index=df.index)

	# Make price label columns
	label_cols['ret_simple_oc'] = (df['pba_close'] / df['pba_open']) - 1
	label_cols['ret_simple_oc2'] = (df['pba_close'].shift(periods=-1) / df['pba_open']) - 1
	label_cols['ret_simple_oa'] = (df['pba_avgPrice'] / df['pba_open']) - 1
	label_cols['ret_simple_oa2'] = (df['pba_avgPrice'].shift(periods=-1) / df['pba_open']) - 1
	label_cols['ret_simple_oo'] = df['pba_open'].pct_change().dropna().shift(periods=-1)
	label_cols['ret_simple_cc'] = df['pba_close'].pct_change().dropna().shift(periods=-1)     # full access
	label_cols['ret_simple_aa'] = df['pba_avgPrice'].pct_change().dropna().shift(periods=-1)  # full access
	label_cols['ret_simple_hl'] = (df['pba_high'] / df['pba_low']) - 1

	label_cols['ret_dir_oc'] = np.sign(label_cols['ret_simple_oc'])
	label_cols['ret_dir_oc2'] = np.sign(label_cols['ret_simple_oc2'])
	label_cols['ret_dir_oa'] = np.sign(label_cols['ret_simple_oa'])
	label_cols['ret_dir_oa2'] = np.sign(label_cols['ret_simple_oa2'])
	label_cols['ret_dir_oo'] = np.sign(label_cols['ret_simple_oo'])
	label_cols['ret_dir_cc'] = np.sign(label_cols['ret_simple_cc']) # full access
	label_cols['ret_dir_aa'] = np.sign(label_cols['ret_simple_aa']) # full access

	# Make volatility label columns
	label_cols['ret_vol_simple_oc'] = (df['vol_close'] / df['vol_open']) - 1
	label_cols['ret_vol_simple_oc2'] = (df['vol_close'].shift(periods=-1) / df['vol_open']) - 1
	label_cols['ret_vol_simple_oa'] = (df['vol_avgPrice'] / df['vol_open']) - 1
	label_cols['ret_vol_simple_oa2'] = (df['vol_avgPrice'].shift(periods=-1) / df['vol_open']) - 1
	label_cols['ret_vol_simple_oo'] = df['vol_open'].pct_change().dropna().shift(periods=-1)
	label_cols['ret_vol_simple_cc'] = df['vol_close'].pct_change().dropna().shift(periods=-1)     # full access
	label_cols['ret_vol_simple_aa'] = df['vol_avgPrice'].pct_change().dropna().shift(periods=-1)  # full access
	label_cols['ret_vol_simple_hl'] = (df['vol_high'] / df['vol_low']) - 1

	label_cols['ret_vol_dir_oc'] = np.sign(label_cols['ret_vol_simple_oc'])
	label_cols['ret_vol_dir_oc2'] = np.sign(label_cols['ret_vol_simple_oc2'])
	label_cols['ret_vol_dir_oa'] = np.sign(label_cols['ret_vol_simple_oa'])
	label_cols['ret_vol_dir_oa2'] = np.sign(label_cols['ret_vol_simple_oa2'])
	label_cols['ret_vol_dir_oo'] = np.sign(label_cols['ret_vol_simple_oo'])
	label_cols['ret_vol_dir_cc'] = np.sign(label_cols['ret_vol_simple_cc']) # full access
	label_cols['ret_vol_dir_aa'] = np.sign(label_cols['ret_vol_simple_aa']) # full access

	return label_cols


if __name__ == '__main__':
	main(sys.argv[1:])
