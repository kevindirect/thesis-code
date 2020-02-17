"""
Kevin Patel
"""
import sys
from os.path import basename
import logging

from common_util import RAW_DIR, DT_HOURLY_FREQ, benchmark, get_cmd_args, isnt, load_json, dti_local_time_mask
from raw.common import GMT_OFFSET_COL_SFX, default_row_masksfile
from data.data_api import DataAPI
from data.data_util import make_entry


def dump_row_masks(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	cmd_arg_list = ['row_masksfile=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	row_masksfile = default_row_masksfile if (isnt(cmd_input['row_masksfile='])) else cmd_input['row_masksfile=']
	row_masks = load_json(row_masksfile, dir_path=RAW_DIR)

	for keychain, raw_rec, raw_df in DataAPI.axe_yield(['root', 'root_split_rows'], lazy=False):
		logging.info(str(keychain))
		asset_name, data_subset = keychain[0], keychain[-1]

		for mask_type, mask in row_masks[asset_name][data_subset].items():
			logging.info('mask type: {}'.format(mask_type))
			mask_freq = DT_HOURLY_FREQ if (mask_type == 'hrm') else None
			mask_df = dti_local_time_mask(raw_df.dropna(how='all').index, mask['interval'], mask['tz'])
			logging.debug(mask_df)
			DataAPI.dump(make_entry('raw', mask_type, data_subset, mask_freq, base_rec=raw_rec), mask_df)
			logging.info('dumped {} {}...'.format(mask_type, data_subset))


if __name__ == '__main__':
	with benchmark('ttf') as b:
		with DataAPI(async_writes=True):
			dump_row_masks(sys.argv[1:])
