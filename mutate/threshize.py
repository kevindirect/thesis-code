# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize

from common_util import DT_CAL_DAILY_FREQ, benchmark, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.thresh import *


def threshize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	# Get col subsetters
	pba = chained_filter(dfs['sp_500_raw_1'].columns, [cs['#pba']['ohlc']])
	vol = chained_filter(dfs['sp_500_raw_1'].columns, [cs['#vol']['ohlc']])
	pba_vol = pba + vol

	search_terms = {
		'stage': 'raw'
	}
	date_range = {
		'id': ('lt', 2018)
	}
	dfs, recs = {}, {}
	for rec, df in DataAPI.generate(search_terms):
		recs[rec.name] = rec
		dfs[rec.name] = df.loc[search_df(df, date_range)]

	# PRICE VOL BASED THRESH
	source_pfx = ['pba', 'vol']
	fs_pairs = [
		('close', 'open'),
		('high', 'low'),
		('avgPrice', 'open'),
		('close', 'avgPrice')
	]
	fs_singles = ['open', 'high', 'low', 'close', 'avgPrice']

	# SENT BASED THRESH


	# CONSTANT THRESH
	start = 1e-7
	end = .1
	pct_threshes = np.linspace(start, end, 100)

	# Processing loop
	for df_name, df_whole in dfs.items():
		original = recs[rec.name]
		df = df_whole[pba_vol]

		entry = {
			'freq': original.freq,
			'root': original.root,
			'basis': original.name,
			'stage': 'mutate',
			'mutate_type': 'thresh',
			'hist': original.hist +'raw->mutate_thresh',
			'desc': 'ansr threshold'
		}

		# Magic happens here
	
		print('\tdumping', end='...', flush=True)
		DataAPI.dump(joined, entry)
		print('done')
	
if __name__ == '__main__':
	threshize(sys.argv[1:])
