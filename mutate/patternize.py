# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
# from pandas.testing import assert_frame_equal
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.preprocessing import Binarizer
# from sklearn.decomposition import PCA
from numba import jit, vectorize
from dask import delayed, compute

from common_util import DT_CAL_DAILY_FREQ, benchmark, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.encoders import *


def patternize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	search_terms = {
		'stage': 'raw',
		'root': 'sp_500',
		'basis': 'sp_500'
	}
	date_end = {
		'id': ('lt', 2018)
	}
	dfs = {}
	for rec, df in DataAPI.generate(search_terms):
		logging.info(rec.name)
		dfs[rec.name] = df.loc[search_df(df, date_end)]

	# df = dd.from_pandas(dfs['sp_500_raw_1'])
	pba = chained_filter(df.columns, [cs['#pba']['ohlc']])
	vol = chained_filter(df.columns, [cs['#vol']['ohlc']])
	pba_vol = pba + vol

	pv = df.loc[:, pba_vol]

	with benchmark('standard') as b:
		std_result = sac(pv, diff_change)
		# std_result = (pv.pipe(sac, day_change)	
		# 			.pipe(sac, np.sign, gb=None))

	# with benchmark('dask') as b:
	# 	result0 = parsac(pv, day_change)
	# 	result1 = parsac(result0, np.sign, gb=None)
	# 	dk_result = result1.compute()

	# result0.visualize("result0.svg")
	# result1.visualize("result1.svg")
	# dk_result.visualize("dk_result.svg")

	# assert_frame_equal(std_result, dk_result)


	result = {}
	col_names = df.columns

	gb = pd.Grouper(freq=DT_CAL_DAILY_FREQ)
	df = delayed(df.groupby)(gb)

	for col_name in col_names:
		result[col_name] = delayed(day_change)(df[col_name])

	result_a = delayed(compute)(result)
	result_a = delayed(pd.DataFrame.from_dict)(result[0])

	result = {}
	col_names = result_a.columns

	for col_name in col_names:
		result[col_name] = delayed(np.sign)(result_a[col_name])

	result_b = delayed(compute)(result)
	result_b = delayed(pd.DataFrame.from_dict)(result[0])

	result_b.visualize('full.svg')
	res = compute(result_b)
	# result = delayed(compute)(result)


	# print(t1.fit_transform(df[pba_vol].dropna()))
	# pba = chained_filter(df.columns, [cs['#pba']['ohlc']])
	# vol = chained_filter(df.columns, [cs['#vol']['ohlc']])
	# pba_vol = pba + vol
	

	# with benchmark('transforms') as b:
	# 	ve = vol_estimates(df[pba_vol])

	# print(ve)

	
if __name__ == '__main__':
	patternize(sys.argv[1:])
