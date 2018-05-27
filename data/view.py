# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import search_df, get_subset, cust_count, benchmark
from data.common import dum
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs


def view(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	# Stuff
	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'normalize',
		'raw_cat': 'us_equity_index'
	}
	recs = {}
	dfs = {}
	for rec, df in DataAPI.generate(search_terms):
		recs[rec.root] = rec
		dfs[rec.root] = df
	logging.info('thresh data loaded')

	for key in recs.keys():
		df = dfs[key]
		print(recs[key].desc)
		pba_cust, pba_count_df = cust_count(df.loc[:, 'pba_avgPrice'])
		vol_cust, vol_count_df = cust_count(df.loc[:, 'vol_avgPrice'])
		print(pba_count_df.value_counts())
		print(vol_count_df.value_counts())
	


if __name__ == '__main__':
	view(sys.argv[1:])