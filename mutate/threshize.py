# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize

from common_util import DT_CAL_DAILY_FREQ, get_custom_biz_freq, get_missing_dt, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.thresh import get_thresh_fth


def threshize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	# LOAD DATA
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


	# THRESH SPECIFICATION
	# price vol based threshold
	pba = [cs['#pba']['ohlc']]
	vol = [cs['#vol']['ohlc']]
	source_pfx = ['pba', 'vol']
	fs_pairs = [
		('close', 'open'),
		('high', 'low'),
		('avgPrice', 'open'),
		('close', 'avgPrice')
	]
	fs_singles = ['open', 'high', 'low', 'close', 'avgPrice']

	# trmi based threshold
	trmi = [cs['#trmi']['all']]

	# constant percentage threshold
	start = 1e-7
	end = .5
	step = float(start/2)
	pct_threshes = np.arange(start, end+step, step=step)


	# FTH THRESHIZE LOOP
	for df_name, df_whole in dfs.items():
		original = recs[rec.name]
		df = df_whole

		entry = make_thresh_entry('fth spread thresh', 'raw->mutate_thresh', original)

		# price based


		# vol based


		# trmi based


		# constant


		# Magic happens here
		get_thresh_fth(intraday_df, thresh_type='spread', shift=False, org_freq=DT_HOURLY_FREQ, agg_freq=DT_BIZ_DAILY_FREQ, shift_freq=DT_BIZ_DAILY_FREQ, pfx='')



		# DUMP THRESHOLD DFs
		print('\tdumping', end='...', flush=True)
		DataAPI.dump(joined, entry)
		print('done')


def make_thresh_entry(desc, hist, base_rec):
	return {
		'freq': base_rec.freq,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'thresh',
		'hist': '->'.join([base_rec.hist, hist]),
		'desc': desc
	}


if __name__ == '__main__':
	threshize(sys.argv[1:])
