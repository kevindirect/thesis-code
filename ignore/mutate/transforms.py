# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from numba import jit, vectorize
from dask import delayed

from common_util import search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.ops import *


def transforms(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	search_terms = {
		'stage': 'raw',
		'root': 'russell_2000',
		'basis': 'russell_2000'
	}
	date_end = {
		'id': ('lt', 2018)
	}
	dfs = {}
	for rec, df in DataAPI.generate(search_terms):
		logging.info(rec.name)
		dfs[rec.name] = df.loc[search_df(df, date_end)]

	df = dfs['russell_2000_raw_3']
	pba = chained_filter(df.columns, [cs['#pba']['ohlc']])
	vol = chained_filter(df.columns, [cs['#vol']['ohlc']])
	pba_vol = pba + vol

	print(t1.fit_transform(df[pba_vol].dropna()))
	# pba = chained_filter(df.columns, [cs['#pba']['ohlc']])
	# vol = chained_filter(df.columns, [cs['#vol']['ohlc']])
	# pba_vol = pba + vol

	# with benchmark('transforms') as b:
	# 	ve = vol_estimates(df[pba_vol])

	# print(ve)

t1 = make_pipeline(Binarizer(), PCA())
	
if __name__ == '__main__':
	transforms(sys.argv[1:])
