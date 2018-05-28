# Kevin Patel

import sys
import os
from functools import partial
from itertools import product
import logging

import numpy as np
import pandas as pd
from scipy.stats import zscore

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, load_json, get_custom_biz_freq, outer_join, right_join, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import default_pattern_threshfile

def normalize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
	pattern_threshfile = default_pattern_threshfile
	pattern_info = load_json(pattern_threshfile, dir_path=MUTATE_DIR)

	# Embargo 2018 data (final validation)
	date_range = {
		'id': ('lt', 2018)
	}

	# raw sourced price and vol series
	search_terms = {
		'stage': 'raw',
		'raw_cat': 'us_equity_index'
	}
	raw_pba_dfs, raw_vol_dfs, raw_recs = {}, {}, {}
	for rec, raw_df in DataAPI.generate(search_terms):
		raw_pba_cols = chained_filter(raw_df.columns, [cs['#pba']['ohlc']])
		raw_vol_cols = chained_filter(raw_df.columns, [cs['#vol']['ohlc']])
		raw_pba_dfs[rec.root] = raw_df.loc[search_df(raw_df, date_range), raw_pba_cols]
		raw_vol_dfs[rec.root] = raw_df.loc[search_df(raw_df, date_range), raw_vol_cols]
		raw_recs[rec.root] = rec
	logging.info('raw data loaded')

	# mutate thresh sourced return series
	thresh_pba_vol_cols = get_thresh_pattern_series(pattern_info)
	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'thresh',
		'raw_cat': 'us_equity_index'
	}
	thresh_dfs, thresh_recs = {}, {}
	for rec, thresh_df in DataAPI.generate(search_terms):
		thresh_dfs[rec.root] = thresh_df.loc[search_df(thresh_df, date_range), thresh_pba_vol_cols]
		thresh_recs[rec.root] = rec
	logging.info('thresh data loaded')

	assert(raw_recs.keys() == thresh_recs.keys())

	for root_name in raw_recs:
		logging.info('asset: ' +str(root_name))

		# PBA
		raw_pba_dzn_df = dayznorm(raw_pba_dfs[root_name])
		raw_pba_dzn_entry = make_normalize_entry('raw_pba_dzn', 'mutate_normalize', raw_recs[root_name])
		logging.debug('dumping raw_pba normalized df ' +str(raw_pba_dzn_entry['desc']) +'...')
		DataAPI.dump(raw_pba_dzn_df, raw_pba_dzn_entry)

		raw_pba_dmx_df = dayznorm(raw_pba_dfs[root_name])
		raw_pba_dmx_entry = make_normalize_entry('raw_pba_dmx', 'mutate_normalize', raw_recs[root_name])
		logging.debug('dumping raw_pba normalized df ' +str(raw_pba_dmx_entry['desc']) +'...')
		DataAPI.dump(raw_pba_dmx_df, raw_pba_dmx_entry)

		# VOL
		raw_vol_dzn_df = dayznorm(raw_vol_dfs[root_name])
		raw_vol_dzn_entry = make_normalize_entry('raw_vol_dzn', 'mutate_normalize', raw_recs[root_name])
		logging.debug('dumping raw_vol normalized df ' +str(raw_vol_dzn_entry['desc']) +'...')
		DataAPI.dump(raw_vol_dzn_df, raw_vol_dzn_entry)

		raw_vol_dmx_df = dayznorm(raw_vol_dfs[root_name])
		raw_vol_dmx_entry = make_normalize_entry('raw_vol_dmx', 'mutate_normalize', raw_recs[root_name])
		logging.debug('dumping raw_vol normalized df ' +str(raw_vol_dmx_entry['desc']) +'...')
		DataAPI.dump(raw_vol_dmx_df, raw_vol_dmx_entry)

		# THRESH
		thresh_dzn_df = dayznorm(thresh_dfs[root_name])
		thresh_dzn_entry = make_normalize_entry('thresh_dzn', 'mutate_normalize', thresh_recs[root_name])
		logging.debug('dumping thresh normalized df ' +str(thresh_dzn_entry['desc']) +'...')
		DataAPI.dump(thresh_dzn_df, thresh_dzn_entry)

		thresh_dmx_df = dayznorm(thresh_dfs[root_name])
		thresh_dmx_entry = make_normalize_entry('thresh_dmx', 'mutate_normalize', thresh_recs[root_name])
		logging.debug('dumping thresh normalized df ' +str(thresh_dmx_entry['desc']) +'...')
		DataAPI.dump(thresh_dmx_df, thresh_dmx_entry)

		DataAPI.update_record()


def get_thresh_pattern_series(pattern_info):
	"""
	Return mapping of return columns to lists of candidate threshold columns.
	"""
	tup_str = lambda c: '_'.join([c[0], c[1], c[2], c[3], sc, c[4]])

	all_src = pattern_info['src']
	fast_slow = pattern_info['fast_slow']
	thresh_type = pattern_info['thresh_type']
	time_hor = pattern_info['time_hor']
	shift_code = pattern_info['shift_code']
	pattern_trans = pattern_info['pattern_trans']
	
	thresh_pattern_cols = []
	for sc in shift_code:
		return list(map(tup_str, product(all_src, fast_slow, thresh_type, time_hor, pattern_trans[sc])))

def dayznorm(df):
	cust = get_custom_biz_freq(df)
	return df.groupby(pd.Grouper(freq=cust)).transform(zscore)

def dayminmaxnorm(df):
	cust = get_custom_biz_freq(df)
	bipolar_mm_transform = lambda ser: 2 * ((ser-ser.min()) / (ser.max()-ser.min())) - 1
	return df.groupby(pd.Grouper(freq=cust)).transform(bipolar_mm_transform)

def make_normalize_entry(desc, hist, base_rec):
	prev_hist = '' if isinstance(base_rec.hist, float) else str(base_rec.hist)

	return {
		'freq': DT_BIZ_DAILY_FREQ,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'normalize',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([prev_hist, hist]),
		'desc': desc
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		normalize(sys.argv[1:])
