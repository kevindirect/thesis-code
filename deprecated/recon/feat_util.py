"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import remove_dups_list, list_get_dict, get_nmost_nulled_cols_df
from recon.common import dum


def split_ser(ser):
	"""
	Split a comma separated series of string values into a dataframe
	"""
	list_ser = ser.str.split(',', n=-1, expand=False)
	split_df = list_ser.apply(pd.Series)
	return split_df

def handle_nans_df(df, method='drop_row', max_col_drop=1):
	# TODO - add in threshold
	# TODO - add in a method to ffill if under threshold
	if (method == 'drop_row'):
		return df.dropna(axis=0, how='any')
	elif (method == 'drop_col'):
		drop_cols = get_nmost_nulled_cols_df(df, n=max_col_drop)
		return df.drop(labels=drop_cols, axis=1)

def split_cluster_ser(ser, sklearn_cluster, col_pfx=None):
	col_pfx = ser.name if (col_pfx is None) else col_pfx
	sax_df = handle_nans_df(split_ser(ser)).add_prefix(col_pfx)
	clustered_values = sklearn_cluster.fit(sax_df.values).labels_
	clustered = pd.Series(data=clustered_values, index=sax_df.index)

	return clustered

def gen_cluster_feats(feat_dict, feat_paths, asset_name, cluster_info):
	for feat_path in filter(lambda feat_path: feat_path[0]==asset_name, feat_paths):
		cname_pfx = '_'.join(feat_path[:0:-1] + [cluster_info['sfx']]) + '_'
		logging.info(cname_pfx)
		feat_df = list_get_dict(feat_dict, feat_path) \
			.apply(split_cluster_ser, axis=0, sklearn_cluster=cluster_info['cl']) \
			.add_prefix(cname_pfx)

		yield feat_df

def gen_split_feats(feat_dict, feat_paths, asset_name):
	for feat_path in filter(lambda feat_path: feat_path[0]==asset_name, feat_paths):
		cname_pfx = feat_path[-1] + '_'
		# cname_pfx = '_'.join(feat_path[:0:-1]) + '_'
		logging.info(cname_pfx[:-1])

		feat_df = list_get_dict(feat_dict, feat_path)
		for feat_name in feat_df:
			yield handle_nans_df(split_ser(feat_df[feat_name])) \
				.add_prefix(cname_pfx +feat_name +'_') \
				.transform(pd.to_numeric) \
