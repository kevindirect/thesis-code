# Kevin Patel

import sys
import os
from itertools import product
import logging

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from numba import jit, vectorize
# from dask import delayed, compute

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, load_json, get_custom_biz_freq, flatten2D, outer_join, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum, default_threshfile
from mutate.label import *


def labelize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
	threshfile = default_threshfile

	thresh_info = load_json(threshfile, dir_path=MUTATE_DIR)

	# search_terms = {
	# 	'stage': 'mutate',
	# 	'mutate_type': 'thresh',
	# 	'raw_cat': 'us_equity_index'
	# }
	# thresh_dfs = {}
	# for rec, thresh_df in DataAPI.generate(search_terms):
	# 	thresh_dfs[rec.root] = thresh_df.loc[search_df(thresh_df, date_range)]
	# logging.info('thresh data loaded')

	ret_groups = get_ret_groups(thresh_info)

	for ret_col in ret_groups:
		print(ret_col)
		print(len(ret_groups[ret_col]))

		# intraday_triple_barrier(intraday_df, col_name, scalar={'up': .55, 'down': .45}, agg_freq=DT_BIZ_DAILY_FREQ):
		# Return modifiers:
		# 	- signed / absolute value

		# Threshold modifiers:
		# 	- shifted / unshifted
		#	- up/down scalars

		# Other thresholds:
		#	- Constant percent threshold



	# for name in price_dfs.keys():
	# 	logging.info('asset:', name)

		# Make dict of {return_series: [thresh_series, ...]}

			# Ennummerate all possible return series to threshold on

			# For each return series determine all possible thresholds

		# For each key in dict

			# New DF

			# compute labels and add as a column

			# Dump DF


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

def get_ret_groups(thresh_info):
	"""
	Return mapping of return columns to lists of candidate threshold columns.
	"""
	def get_thresh_cols(scg, scode, tmh, trans):
		return ['_'.join([scg, thresh_combo[0], scode, thresh_combo[1]]) for thresh_combo in product(tmh, trans)]

	src = thresh_info['src']
	fast_slow = thresh_info['fast_slow']
	thresh_type = thresh_info['thresh_type']
	time_hor = thresh_info['time_hor']
	shift_code = thresh_info['shift_code']
	all_trans = thresh_info['all_trans']
	ret_trans = thresh_info['ret_trans']
	# oth_trans = {sc: list(filter(lambda x: x not in ret_trans[sc], all_trans[sc])) for sc in all_trans.keys()}

	ret_groups = {}
	src_groups = ['_'.join(combo) for combo in product(src, fast_slow, thresh_type)]

	for src_group in src_groups:
		for sc in shift_code:
			for ret_combo in product(time_hor, ret_trans[sc]):
				ret_col = '_'.join([src_group, ret_combo[0], sc, ret_combo[1]])
				ret_groups[ret_col] = flatten2D([get_thresh_cols(src_group, sc_2, time_hor, all_trans[sc_2]) for sc_2 in shift_code])
				ret_groups[ret_col].remove(ret_col)

	return ret_groups


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		labelize(sys.argv[1:])