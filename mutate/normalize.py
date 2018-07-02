# Kevin Patel

import sys
import os
import logging

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, list_get_dict, is_empty_df, search_df, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from mutate.common import dum
from mutate.pattern_util import day_norm, NORM_FUN_MAP


def normalize(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

	# Embargo 2018 data (final validation)
	date_range = {
		'id': ('lt', 2018)
	}
	pre_norm_paths, pre_norm_recs, pre_norm_dfs = DataAPI.load_from_dg(dg['normalize']['pre_norm'], cs2['normalize']['pre_norm'])

	for key_chain in pre_norm_paths:
		logging.info('asset: ' +key_chain[0])
		pre_norm_rec = list_get_dict(pre_norm_recs, key_chain)
		pre_norm_df = list_get_dict(pre_norm_dfs, key_chain)
		pre_norm_df = pre_norm_df.loc[search_df(pre_norm_df, date_range), :]

		for norm_code, norm_fun in NORM_FUN_MAP.items():
			normed_df = day_norm(pre_norm_df, norm_fun, freq=DT_CAL_DAILY_FREQ).dropna(axis=0, how='all')
			desc = '_'.join([key_chain[-1], norm_code])
			entry = make_normalize_entry(desc, pre_norm_rec)

			assert(not is_empty_df(normed_df))
			logging.debug('dumping ' +desc +'...')
			DataAPI.dump(normed_df, entry)

	DataAPI.update_record() # Sync


def make_normalize_entry(desc, base_rec):
	prev_hist = '' if isinstance(base_rec.hist, float) else str(base_rec.hist)

	return {
		'freq': DT_HOURLY_FREQ,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'normalize',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([prev_hist, 'mutate_normalize']),
		'desc': desc
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		normalize(sys.argv[1:])
