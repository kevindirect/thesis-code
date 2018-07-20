# Kevin Patel

import sys
import getopt
import logging

from common_util import RAW_DIR, DT_HOURLY_FREQ, load_json, load_df, series_to_dti, right_join, outer_join, list_get_dict, get_time_mask
from raw.common import GMT_OFFSET_COL_SFX, default_row_masksfile
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2


def dump_intraday_row_masks(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	usage = lambda: print('dump_id_rm.py [-m <row_masksfile>]')
	row_masksfile = default_row_masksfile

	try:
		opts, args = getopt.getopt(argv, 'hm:', ['help', 'row_masksfile='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-m', '--row_masksfile'):
			row_masksfile = arg

	row_masks = load_json(row_masksfile, dir_path=RAW_DIR)

	all_raw = ['raw', 'all']
	raw_dg, raw_cs = list_get_dict(dg, all_raw), list_get_dict(cs2, all_raw)
	raw_paths, raw_recs, raw_dfs = DataAPI.load_from_dg(raw_dg, raw_cs, subset=['raw_pba', 'raw_vol'])

	for key_chain in raw_paths:
		asset_name, data_subset = key_chain[0], key_chain[-1]
		raw_rec, raw_df = list_get_dict(raw_recs, key_chain), list_get_dict(raw_dfs, key_chain)
		gmt_col = '_'.join([data_subset[-3:], GMT_OFFSET_COL_SFX])
		logging.info(asset_name)

		for mask_type, mask in row_masks[asset_name][data_subset].items():
			logging.info('mask name: ' +str(mask_type))
			mask_df = get_time_mask(raw_df, offset_col_name=gmt_col, offset_tz=mask['target_tz'], time_range=mask['time_range'])
			desc = '_'.join([data_subset, mask['desc_sfx']])
			entry = make_id_rm_entry(desc, raw_rec)
			logging.info('dumping ' +str(desc) +'...')
			logging.debug(mask_df)
			DataAPI.dump(mask_df, entry)

	DataAPI.update_record()

def make_id_rm_entry(desc, base_rec):
	prev_hist = '' if isinstance(base_rec.hist, float) else str(base_rec.hist)
	mutate_type = 'intraday_row_mask'

	return {
		'freq': DT_HOURLY_FREQ,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'raw',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([prev_hist, str('raw_' +desc)]),
		'desc': desc
	}

if __name__ == '__main__':
	dump_intraday_row_masks(sys.argv[1:])
