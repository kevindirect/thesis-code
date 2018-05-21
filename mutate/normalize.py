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
	thresh_pba_vol_cols = get_thresh_pattern_series(pattern_info)

	# Embargo 2018 data (final validation)
	date_range = {
		'id': ('lt', 2018)
	}

	# raw sourced price series
	search_terms = {
		'stage': 'raw',
		'raw_cat': 'us_equity_index'
	}
	price_dfs, price_recs = {}, {}
	for rec, price_df in DataAPI.generate(search_terms):
		price_cols = chained_filter(price_df.columns, [cs['#pba']['ohlc']]) + chained_filter(price_df.columns, [cs['#vol']['ohlc']])
		price_dfs[rec.root] = price_df.loc[search_df(price_df, date_range), price_cols]
		price_recs[rec.root] = rec
	logging.info('price data loaded')

	# mutate thresh sourced return series
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

	assert(price_recs.keys() == thresh_recs.keys())

	for root_name in price_recs:
		logging.info('asset: ' +str(root_name))

		dzn_price = dayznorm(price_dfs[root_name])
		dmx_price = dayminmaxnorm(price_dfs[root_name])
		dzn_thresh = dayznorm(thresh_dfs[root_name])
		dmx_thresh = dayminmaxnorm(thresh_dfs[root_name])

		# Z Score Normalization
		dzn_df = outer_join(dzn_price, dzn_thresh)
		dzn_entry = make_normalize_entry('dzn_price_thresh', price_recs[root_name], thresh_recs[root_name])
		logging.debug('dumping normalized df ' +str(dzn_entry['desc']) +'...')
		DataAPI.dump(dzn_df, dzn_entry)

		# Min Max Normalization
		dmx_df = outer_join(dmx_price, dmx_thresh)
		dmx_entry = make_normalize_entry('dmx_price_thresh', price_recs[root_name], thresh_recs[root_name])
		logging.debug('dumping normalized df ' +str(dmx_entry['desc']) +'...')
		DataAPI.dump(dmx_df, dmx_entry)

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

def make_normalize_entry(desc, first_rec, *args):
	assert(all(entry.root==first_rec.root for entry in args))
	assert(all(entry.raw_cat==first_rec.raw_cat for entry in args))

	names = [rec.name for rec in args]
	combined_basis = '(' +first_rec.name +', ' +', '.join(names) +')'

	hist_mapper = lambda hist: '' if (not isinstance(hist, str)) else hist
	hists = list(map(hist_mapper, [rec.hist for rec in args]))
	combined_hist = '(' +hist_mapper(first_rec.hist) +', ' +', '.join(hists) +')'

	return {
		'freq': DT_BIZ_DAILY_FREQ,
		'root': first_rec.root,
		'basis': combined_basis,
		'stage': 'mutate',
		'mutate_type': 'mutate_normalize',
		'raw_cat': first_rec.raw_cat,
		'hist': '->'.join([combined_hist, 'mutate_normalize']),
		'desc': desc
	}

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		normalize(sys.argv[1:])
