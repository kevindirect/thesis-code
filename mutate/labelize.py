# Kevin Patel

import sys
import os
from itertools import product
import logging

import numpy as np
import pandas as pd

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, load_json, get_custom_biz_freq, flatten2D, outer_join, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum, default_threshfile
from mutate.label import *


def labelize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
	threshfile = default_threshfile

	thresh_info = load_json(threshfile, dir_path=MUTATE_DIR)

	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'thresh',
		'raw_cat': 'us_equity_index'
	}
	thresh_recs = {}
	thresh_dfs = {}
	for rec, thresh_df in DataAPI.generate(search_terms):
		thresh_recs[rec.root] = rec
		thresh_dfs[rec.root] = thresh_df.loc[search_df(thresh_df, date_range)]
	logging.info('thresh data loaded')

	ret_groups = get_ret_groups(thresh_info)


	# ********** VARIABLE THRESHOLD **********
	# Threshold Modifiers:
	# shifted / unshifted
	shiftedness = [True, False]

	# Power of 2 Scaling
	scale_pow_exp = 5
	scalings = [(2**n,) for n in range(-scale_pow_exp, scale_pow_exp+1)]

	# Return Modifiers:
	# signed / absolute value
	signedness = [True, False]


	# ********** CONSTANT THRESHOLD **********
	start = 10**-6
	stop = 10**1
	step = start / 2
	const_threshes = np.arange(start, stop, step)

	for name in thresh_dfs:
		logging.info('asset:', name)
		thresh_df = thresh_dfs[name]

		for ret_col in ret_groups:
			logging.debug('ret_col:', ret_col)
			labels = thresh_df.loc[:, [ret_col]].copy()

			logging.info('random variable threshes')
			for thresh_id, thresh_col in enumerate(ret_groups[ret_col]):
				logging.debug('thresh_col:', thresh_col +'id:', thresh_id)
				ret_thresh_df = thresh_df.loc[:, [ret_col, thresh_col]]

				for shf, scl, sgn in product(shiftedness, scalings, signedness):
					ret_thresh_df = thresh_df.loc[:, [ret_col, thresh_col]]
					shf_str = 'shf' if (shf) else 'uns'
					sgn_str = 'sgn' if (sgn) else 'abs'
					scl_str = '_'.join([str(sclr) for slcr in scl])

					if (shf): # Shifted
						if ('_af_' in ret_col):
							shift_freq = DT_BIZ_DAILY_FREQ,
						elif ('_of_' in ret_col):
							shift_freq = DT_HOURLY_FREQ
						ret_thresh_df[thresh_col] = ret_thresh_df[thresh_col].shift(freq=shift_freq)

					if (sgn): # Unsigned
						ret_thresh_df[thresh_col] = ret_thresh_df[thresh_col].abs()

					itb = intraday_triple_barrier(ret_thresh_df, scalar=scl)
					col_prefix = '_'.join([ret_col, 'rvt', str(thresh_id), shf_str, sgn_str, scl_str])
					labels = outer_join(labels, itb.add_prefix(col_prefix +'_'))

			logging.info('constant percentage threshes')
			for const_thresh in const_threshes:
				logging.debug('pct:', str(const_thresh))
				ret_thresh_df = thresh_df.loc[:, [ret_col]]
				ct_col = '_'.join(['cpt', str(const_thresh)])
				ret_thresh_df[ct_col] = const_thresh
				itb = intraday_triple_barrier(ret_thresh_df, scalar=scl)
				col_prefix = '_'.join([ret_col, ct_col])
				labels = outer_join(labels, itb.add_prefix(col_prefix +'_'))

			logging.info('dumping labels')
			entry = make_label_entry(ret_col, 'mutate_label', thresh_recs[name])
			DataAPI.dump(thresh_df, entry)
			logging.info('done dumping labels')

		DataAPI.update_record()


def make_label_entry(desc, hist, base_rec):
	prev_hist = '' if np.isnan(base_rec.hist) else str(base_rec.hist)

	return {
		'freq': base_rec.freq,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'label',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([prev_hist, hist]),
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