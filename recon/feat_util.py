# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import remove_dups_list, list_get_dict
from recon.common import dum


def split_ser(ser, num_cols, pfx=''):
	split_df = pd.DataFrame(index=ser.index)
	column_names = ['_'.join([pfx, str(i)]) for i in range(num_cols)]
	split_df[column_names] = ser.str.split(',', num_cols, expand=True)
	return split_df

def handle_nans_df(df, method='drop'):
	# TODO - add in threshold
	# TODO - add in a method to ffill if under threshold
	return df.dropna(axis=0, how='any')

def split_cluster_ser(ser, sklearn_cluster, col_pfx=None):
	col_pfx = ser.name if (col_pfx is None) else col_pfx
	sax_df = handle_nans_df(split_ser(ser, 8, pfx=col_pfx))
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
