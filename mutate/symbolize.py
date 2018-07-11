# Kevin Patel

import sys
import os
import getopt
import logging

import numpy as np
import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, get_custom_biz_freq, list_get_dict, left_join, outer_join, wrap_parens, search_df, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2, col_subsetters as cs
from mutate.common import STANDARD_DAY_LEN, default_num_sym
from mutate.pattern_util import BREAKPOINT_MAP, encode_df


def symbolize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	usage = lambda: print('symbolize.py [-a <num_sym>]')
	num_sym = default_num_sym

	try:
		opts, args = getopt.getopt(argv, 'ha:', ['help', 'num_sym='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-a', '--num_sym'):  num_sym = int(arg)

	NORM_BREAK_MAP = {
		"dzn": "gaussian",
		"dmx": "uniform"
	}

	# Embargo 2018 data (final validation)
	date_range = {
		'id': ('lt', 2018)
	}
	pre_sym_paths, pre_sym_recs, pre_sym_dfs = DataAPI.load_from_dg(dg['sym']['pre_sym'], cs2['sym']['pre_sym'])

	for key_chain in pre_sym_paths:
		logging.info('asset: ' +key_chain[0])
		logging.info(key_chain[1])
		pre_sym_rec = list_get_dict(pre_sym_recs, key_chain)
		pre_sym_df = list_get_dict(pre_sym_dfs, key_chain)
		pre_sym_df = pre_sym_df.loc[search_df(pre_sym_df, date_range), :]

		break_type = NORM_BREAK_MAP[key_chain[1]]
		sym_df = encode_df(pre_sym_df, BREAKPOINT_MAP[break_type], num_sym, numeric_symbols=True)

		if (logging.getLogger().isEnabledFor(logging.DEBUG)):
			for col_name in sym_df.columns:
				before = pre_sym_df[[col_name]].rename(columns={col_name: str(col_name+'_before')})
				after = sym_df[[col_name]].rename(columns={col_name: str(col_name+'_after')})
				before_after = left_join(before, after).dropna(axis=0, how='any')
				logging.debug(str(before_after))

		desc = break_type[:3] +wrap_parens(str(num_sym))
		sym_entry = make_symbolize_entry(desc, str('mutate_' +desc), pre_sym_rec)
		logging.info('dumping ' +pre_sym_rec.desc +'_' +desc +'...')
		DataAPI.dump(sym_df, sym_entry)

	DataAPI.update_record() # Sync


def make_symbolize_entry(desc, hist, base_rec):
	prev_hist = '' if isinstance(base_rec.hist, float) else str(base_rec.hist)

	return {
		'freq': DT_HOURLY_FREQ,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'symbolize',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([prev_hist, hist]),
		'desc': '_'.join([base_rec.desc, desc])
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		symbolize(sys.argv[1:])
