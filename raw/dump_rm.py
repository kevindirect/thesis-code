"""
Kevin Patel
"""
import sys
from os.path import basename
import logging

from common_util import RAW_DIR, DT_HOURLY_FREQ, get_cmd_args, isnt, load_json, load_df, series_to_dti, right_join, outer_join, list_get_dict, get_time_mask
from raw.common import GMT_OFFSET_COL_SFX, default_row_masksfile
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters as cs


def dump_row_masks(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	cmd_arg_list = ['row_masksfile=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	row_masksfile = default_row_masksfile if (isnt(cmd_input['row_masksfile='])) else cmd_input['row_masksfile=']
	row_masks = load_json(row_masksfile, dir_path=RAW_DIR)

	raw_gmt = ['raw', 'raw_gmtoffset']
	raw_dg, raw_cs = list_get_dict(dg, raw_gmt), list_get_dict(cs, raw_gmt)
	raw_paths, raw_recs, raw_dfs = DataAPI.load_from_dg(raw_dg, raw_cs, subset=['pba', 'vol', 'trmi2', 'trmi3'])

	for key_chain in raw_paths:
		logging.info(str(key_chain))
		asset_name, data_subset = key_chain[0], key_chain[-1]
		raw_rec, raw_df = list_get_dict(raw_recs, key_chain), list_get_dict(raw_dfs, key_chain)
		gmt_col = raw_df.columns[0]
		assert(gmt_col.endswith(GMT_OFFSET_COL_SFX))
		if (raw_df.shape[1] > 1):
			# This method may lead to error if the time_range includes the times when Daylight savings is switched on/off
			raw_df = raw_df.fillna(method='ffill', axis=0)
			logging.info('Found more than one column, ffilled null values')
		logging.debug(raw_df)

		for mask_type, mask in row_masks[asset_name][data_subset].items():
			logging.info('mask name: {}'.format(mask_type))
			mask_df = get_time_mask(raw_df, offset_col_name=gmt_col, offset_tz=mask['target_tz'], time_range=mask['time_range'])
			desc = '_'.join([data_subset, mask['type']])
			entry = make_rm_entry(mask['type'], desc, raw_rec)
			logging.debug(mask_df)
			DataAPI.dump(mask_df, entry)
			logging.info('dumped {}...'.format(desc))
	DataAPI.update_record()

def make_rm_entry(rawtype, desc, base_rec):
	prev_hist = '' if isinstance(base_rec.hist, float) else str(base_rec.hist)

	return {
		'freq': DT_HOURLY_FREQ,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'raw',
		'type': rawtype,
		'cat': base_rec.cat,
		'hist': '->'.join([prev_hist, str(desc)]),
		'desc': desc
	}

if __name__ == '__main__':
	dump_row_masks(sys.argv[1:])
