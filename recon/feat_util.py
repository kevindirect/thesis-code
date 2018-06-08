# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_util import remove_dups_list
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

def split_cluster_ser(ser, sklearn_cluster, col_name_pfx='', cluster_sfx=''):
	col_pfx = '_'.join([col_name_pfx, ser.name])
	logging.info(col_pfx)

	sax_df = handle_nans_df(split_ser(ser, 8, pfx=col_pfx))
	clustered_values = sklearn_cluster.fit(sax_df.values).labels_
	clustered = pd.Series(data=clustered_values, name='_'.join([col_pfx, cluster_sfx]), index=sax_df.index)

	return clustered
	# temp_df = inner_join(label_fct_shf_df, sax_df)
	# feats_only = temp_df[temp_df.columns[1:]]