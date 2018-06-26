# Kevin Patel

import sys
import os
from itertools import product
from functools import reduce
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, load_json, get_custom_biz_freq_ser, flatten2D, left_join, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum, default_label_threshfile, default_labelfile
from mutate.label import *


def labelize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
	label_threshfile = default_label_threshfile
	labelfile = default_labelfile

	thresh_info = load_json(label_threshfile, dir_path=MUTATE_DIR)
	label_info = load_json(labelfile, dir_path=MUTATE_DIR)
	ret_groups = get_ret_groups(thresh_info)

	date_range = {
		'id': ('lt', 2018)
	}
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

	# label_tups = {}
	# for root_name in thresh_recs:
	# 	logging.info('asset: ' +str(root_name))
	# 	rec_labs = make_labels(thresh_dfs[root_name], thresh_recs[root_name], ret_groups, label_info)

	# 	logging.info('dumping labels...')
	# 	for i, rec, lab_df in enumerate(rec_labs):
	# 		logging.debug('dumping label df ' +str(i) +'...')
	# 		DataAPI.dump(lab_df, rec)

	# 	DataAPI.update_record()

	for root_name in thresh_recs:
		logging.info('asset: ' +str(root_name))

		for entry, lab_df in gen_labels(thresh_dfs[root_name], thresh_recs[root_name], ret_groups, label_info):
			logging.debug('dumping label df ' +str(entry['desc']) +'...')
			DataAPI.dump(lab_df, entry)

		DataAPI.update_record()


def get_ret_groups(thresh_info, thresh_src='all'):
	"""
	Return mapping of return columns to lists of candidate threshold columns.
	"""
	def get_thresh_cols(sc, fstg, scode, tmh, trans):
		return ['_'.join([t_combo[0], fstg, t_combo[1], scode, t_combo[2]]) for t_combo in product(sc, tmh, trans)]

	all_src = thresh_info['src']
	fast_slow = thresh_info['fast_slow']
	thresh_type = thresh_info['thresh_type']
	time_hor = thresh_info['time_hor']
	shift_code = thresh_info['shift_code']
	all_trans = thresh_info['all_trans']
	ret_trans = thresh_info['ret_trans']
	oth_trans = {sc: list(filter(lambda x: x not in ret_trans[sc], all_trans[sc])) for sc in all_trans.keys()}

	ret_groups = {}
	fst_groups = ['_'.join(combo) for combo in product(fast_slow, thresh_type)]

	for data_src, fst_group in product(all_src, fst_groups):
		if (thresh_src == 'all'):
			thresh_src = all_src
		elif (thresh_src == 'same'):
			thresh_src = [data_src]

		for sc in shift_code:
			for ret_combo in product(time_hor, ret_trans[sc]):
				ret_col = '_'.join([data_src, fst_group, ret_combo[0], sc, ret_combo[1]])
				ret_groups[ret_col] = flatten2D([get_thresh_cols(thresh_src, fst_group, sc_2, time_hor, oth_trans[sc_2]) for sc_2 in shift_code])

	return ret_groups


def make_labels(base_df, base_rec, return_groups, label_info):
	"""
	Return list of (rec, label_df) tuples
	"""
	ret_mod = label_info['return_mod']
	rvt = label_info['rvt']['thresh_mod']
	# cpt = label_info['cpt']['thresh_def']
	# const_threshes = np.arange(start=cpt['arange']['start'], stop=cpt['arange']['stop'], step=cpt['arange']['step'])

	rec_lab_tups = []
	for ret_col, sgn in product(return_groups.keys(), ret_mod['signedness']):
		if (sgn and '_lh_' in ret_col):
			continue	# lh is inherently unsigned

		label_df_list = []

		# PREPARE RETURN SERIES
		ret_designator = ret_col if (sgn) else str(ret_col +'_uns')
		ret_df = base_df.loc[:, [ret_col]] if (sgn) else base_df.loc[:, [ret_col]].abs()
		ret_df = ret_df.rename(columns={ret_col: ret_designator})
		logging.info('ret_col: ' +str(ret_designator))

		# RANDOM VARIABLE THRESHOLD
		for thresh_col in return_groups[ret_col]:
			logging.debug('thresh_col: ' +str(thresh_col))
			thresh_df = base_df.loc[:, [thresh_col]]
			thresh_pfx = thresh_col +'_'

			for shf in rvt['shiftedness']:
				if (shf):
					pfx = thresh_pfx +'shf_'
					procedure = list(filter(lambda item: item in ['af', 'of'], thresh_col.split('_')))[0]
					ret_thresh_df = shift_time_series_df(procedure, thresh_df, thresh_col, ret_df)
				else:
					pfx = thresh_pfx
					ret_thresh_df = left_join(ret_df, thresh_df)

				for scl in rvt['scalings']:
					pfx_scl = pfx +str(scl) +'_'
					itb = intraday_triple_barrier(ret_thresh_df, scalar=(scl, scl))
					label_df_list.append(itb.add_prefix(pfx_scl))

		# # CONSTANT PERCENTAGE THRESHOLD
		# for const_thresh in const_threshes:
		# 	pfx = '_'.join(['cpt', str(const_thresh)]) +'_'
		# 	thresh_df = pd.DataFrame(index=ret_df.index)
		# 	thresh_df[pfx] = const_thresh
		# 	ret_thresh_df = left_join(ret_df, thresh_df)

		# 	itb = intraday_triple_barrier(ret_thresh_df)
		# 	label_df_list.append(itb.add_prefix(pfx))

		# JOIN
		entry = make_label_entry(ret_designator, 'mutate_label', base_rec)
		labels_df = reduce(left_join, [ret_df] + label_df_list)
		rec_lab_tups.append((entry, labels_df))

	return rec_lab_tups


def gen_labels(base_df, base_rec, return_groups, label_info):
	"""
	Return list of (rec, label_df) tuples
	"""
	ret_mod = label_info['return_mod']
	rvt = label_info['rvt']['thresh_mod']

	rec_lab_tups = []
	for ret_col, sgn in product(return_groups.keys(), ret_mod['signedness']):
		if (sgn and '_lh_' in ret_col):
			continue	# lh is inherently unsigned

		label_df_list = []

		# PREPARE RETURN SERIES
		ret_designator = ret_col if (sgn) else str(ret_col +'_uns')
		ret_df = base_df.loc[:, [ret_col]] if (sgn) else base_df.loc[:, [ret_col]].abs()
		ret_df = ret_df.rename(columns={ret_col: ret_designator})
		logging.info('ret_col: ' +str(ret_designator))

		# RANDOM VARIABLE THRESHOLD
		for thresh_col in return_groups[ret_col]:
			logging.debug('thresh_col: ' +str(thresh_col))
			thresh_df = base_df.loc[:, [thresh_col]]
			thresh_pfx = thresh_col +'_'

			for shf in rvt['shiftedness']:
				if (shf):
					pfx = thresh_pfx +'shf_'
					procedure = list(filter(lambda item: item in ['af', 'of'], thresh_col.split('_')))[0]
					ret_thresh_df = shift_time_series_df(procedure, thresh_df, thresh_col, ret_df)
				else:
					pfx = thresh_pfx
					ret_thresh_df = left_join(ret_df, thresh_df)

				for scl in rvt['scalings']:
					pfx_scl = pfx +str(scl) +'_'
					itb = intraday_triple_barrier(ret_thresh_df, scalar=(scl, scl))
					label_df_list.append(itb.add_prefix(pfx_scl))

		# JOIN
		entry = make_label_entry(ret_designator, 'mutate_label', base_rec)
		labels_df = reduce(left_join, [ret_df] + label_df_list)
		yield (entry, labels_df)


def shift_time_series_df(shift_procedure, to_shift_df, shift_col_name, to_join_df):
	"""
	Return pd.DataFrame of threshold column shifted according to shift_procedure and joined with to_join_df.
	"""
	agg_freq = get_custom_biz_freq_ser(to_shift_df[shift_col_name])

	if (shift_procedure == 'af'):		# Shift by aggregation frequency
		shift_df = to_shift_df[[shift_col_name]].shift(periods=1, freq=agg_freq, axis=0)
		ret_thresh_df = left_join(to_join_df, shift_df)
		ret_thresh_df[shift_col_name] = ret_thresh_df[shift_col_name].fillna(method='ffill')

	elif (shift_procedure == 'of'):		# Shift by original frequency
		shift_df = to_shift_df[[shift_col_name]].groupby(pd.Grouper(freq=agg_freq)).shift(periods=1, axis=0)
		ret_thresh_df = left_join(to_join_df, shift_df)

	return ret_thresh_df


def make_label_entry(desc, hist, base_rec):

	return {
		'freq': DT_BIZ_DAILY_FREQ,
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