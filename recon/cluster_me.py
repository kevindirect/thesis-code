# Kevin Patel

import sys
import os
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize

from common_util import search_df, get_subset, benchmark
from data.data_api import DataAPI
from recon.common import dum



def cluster_me(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	date_range = {
		'id': ('lt', 2018)
	}
	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'saxify',
		'raw_cat': 'us_equity_index'
	}
	sax_recs, sax_dfs = defaultdict(dict), defaultdict(dict)
	for rec, sax_df in DataAPI.generate(search_terms):
		sax_dfs[rec.root][rec.desc] = sax_df.loc[search_df(sax_df, date_range)]
		sax_recs[rec.root][rec.desc] = rec
	logging.info('normalize data loaded')

	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'label',
		'raw_cat': 'us_equity_index'
	}
	lab_recs, lab_dfs = defaultdict(dict), defaultdict(dict)
	for rec, lab_df in DataAPI.generate(search_terms):
		if (rec.desc[:3] == 'pba'):
			lab_dfs[rec.root][rec.desc] = lab_df.loc[search_df(lab_df, date_range)]
			lab_recs[rec.root][rec.desc] = rec
	logging.info('pba label data loaded')

	for root_name in sax_recs:
		logging.info('asset: ' +str(root_name))

		sax_feat_dfs = sax_dfs[root_name]
		sax_lab_dfs = lab_dfs[root_name]

		print(sax_lab_dfs.keys())

		# split_up = pt_df['pba_avgPrice'].str.split(',', num_per-1, expand=True)

	# label encode


	# cluster dimensions

	# convert to cluster

	# shift feature matrix (cluster) up one to next available day

	# select labels

	# get test/train

	# classify


	print(split_up)
	print(split_up[5].value_counts(dropna=False))
	pt_df = pt_df[pt_df['pba_avgPrice'].str.len() == num_per]
	print(split_up)
	# Clustering saxed

	# 

	
if __name__ == '__main__':
	cluster_me(sys.argv[1:])