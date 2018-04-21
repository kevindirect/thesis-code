# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from numba import jit, vectorize
# from dask import delayed, compute

from common_util import DT_HOURLY_FREQ, get_custom_biz_freq, outer_join, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.label import *


def labelize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

	date_range = {
		'id': ('lt', 2018)
	}

	search_terms = {
		'stage': 'raw',
		'raw_cat': 'us_equity_index'
	}
	price_dfs, price_recs = {}, {}
	for rec, df in DataAPI.generate(search_terms):
		price_recs[rec.root] = rec
		price_dfs[rec.root] = df.loc[search_df(df, date_range)]
	logging.info('pricing loaded')

	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'thresh',
		'raw_cat': 'us_equity_index'
	}
	thresh_dfs, thresh_recs = {}, {}
	for rec, df in DataAPI.generate(search_terms):
		thresh_recs[rec.root] = rec
		thresh_dfs[rec.root] = df.loc[search_df(df, date_range)]
	logging.info('threshes loaded')

	for root_eq in price_recs.keys():
		logging.info(root_eq)

		for col_name in thresh_dfs[root_eq].columns:
			print(col_name)
			# labs = intraday_triple_barrier(price_dfs[base], thresh_dfs[base])

		print('#################################')



def make_label_entry(desc, hist, base_rec):
	return {
		'freq': base_rec.freq,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'label',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([str(base_rec.hist), hist]),
		'desc': desc
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		labelize(sys.argv[1:])
		logging.info(str(b))