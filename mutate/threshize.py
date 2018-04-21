# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize

from common_util import DT_HOURLY_FREQ, get_custom_biz_freq, outer_join, right_join, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.thresh import get_thresh_fth


def threshize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

	# LOAD DATA
	search_terms = {
		'stage': 'raw',
		'raw_cat': 'us_equity_index'
	}
	dfs, recs = {}, {}
	for rec, df in DataAPI.generate(search_terms):
		recs[rec.name] = rec
		dfs[rec.name] = df
	logging.info('data loaded')

	# THRESH SPECIFICATION
	# price vol based threshold
	fs_pairs = {
		'oc': ['close', 'open'],
		'lh': ['high', 'low'],
		'oa': ['avgPrice', 'open'],
		'ac': ['close', 'avgPrice']
		# 'oo': ['open', 'open'],
		# 'hh': ['high', 'high'],
		# 'll': ['low', 'low'],
		# 'cc': ['close', 'close'],
		# 'aa': ['avgPrice', 'avgPrice'],
	}

	# trmi based threshold
	trmi = [cs['#trmi']['all']]

	# FTH THRESHIZE LOOP
	logging.info('starting thresh loop')
	date_range = {
		'id': ('lt', 2018)
	}
	for name, whole_df in dfs.items():
		logging.info(name)
		original = recs[name]
		df = whole_df.loc[search_df(df, date_range)]
		thresh_df = pd.DataFrame()

		# trmi_cols = chained_filter(df.columns, [cs['#trmi']['all']])
		# trmi_df = df[trmi_cols]

		# TODO - different files for different fs_pairs AND shift freqs may be a good idea

		# price based
		src = 'pba'
		src_cols = chained_filter(df.columns, [cs['#'+src]['ohlc']])
		src_df = df[src_cols]
		logging.info(src)
		src_fs_pairs = {key: ['_'.join([src, it]) for it in pair] for key, pair in fs_pairs.items()}
		for key, fs_cols in src_fs_pairs.items():
			fs = src_df.loc[:, fs_cols].dropna()
			logging.info(fs_cols)
			custom = get_custom_biz_freq(fs)

			t_type='spread'
			logging.debug(t_type)
			af = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=custom).drop(columns=['fast', 'slow', 'thresh'])
			of = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=DT_HOURLY_FREQ).drop(columns=['fast', 'slow', 'thresh'])
			thresh_df = outer_join(right_join(thresh_df, af), of)

			t_type='ansr'
			logging.debug(t_type)
			af = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=custom).drop(columns=['fast', 'slow', 'thresh'])
			of = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=DT_HOURLY_FREQ).drop(columns=['fast', 'slow', 'thresh'])
			thresh_df = outer_join(outer_join(thresh_df, af), of)

			t_type='log'
			logging.debug(t_type)
			af = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=custom).drop(columns=['fast', 'slow', 'thresh'])
			of = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=DT_HOURLY_FREQ).drop(columns=['fast', 'slow', 'thresh'])
			thresh_df = outer_join(outer_join(thresh_df, af), of)

		# vol based
		src = 'vol'
		src_cols = chained_filter(df.columns, [cs['#'+src]['ohlc']])
		src_df = df[src_cols]
		logging.info(src)
		src_fs_pairs = {key: ['_'.join([src, it]) for it in pair] for key, pair in fs_pairs.items()}
		for key, fs_cols in src_fs_pairs.items():
			fs = src_df.loc[:, fs_cols].dropna()
			logging.info(fs_cols)
			custom = get_custom_biz_freq(fs)

			t_type='spread'
			logging.debug(t_type)
			af = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=custom).drop(columns=['fast', 'slow', 'thresh'])
			of = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=DT_HOURLY_FREQ).drop(columns=['fast', 'slow', 'thresh'])
			thresh_df = outer_join(outer_join(thresh_df, af), of)

			t_type='ansr'
			logging.debug(t_type)
			af = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=custom).drop(columns=['fast', 'slow', 'thresh'])
			of = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=DT_HOURLY_FREQ).drop(columns=['fast', 'slow', 'thresh'])
			thresh_df = outer_join(outer_join(thresh_df, af), of)

			t_type='log'
			logging.debug(t_type)
			af = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=custom).drop(columns=['fast', 'slow', 'thresh'])
			of = get_thresh_fth(fs, thresh_type=t_type, shift=False, src_data_pfx='_'.join([src, key]),
				org_freq=DT_HOURLY_FREQ, agg_freq=custom, shift_freq=DT_HOURLY_FREQ).drop(columns=['fast', 'slow', 'thresh'])
			thresh_df = outer_join(outer_join(thresh_df, af), of)


		# DUMP THRESHOLD DFs
		entry = make_thresh_entry('fth thresh', 'raw->mutate_thresh', original)
		print('\tdumping', end='...', flush=True)
		DataAPI.dump(thresh_df, entry)
		print('done')

	DataAPI.update_record()


def make_thresh_entry(desc, hist, base_rec):
	return {
		'freq': base_rec.freq,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'thresh',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([str(base_rec.hist), hist]),
		'desc': desc
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		threshize(sys.argv[1:])
