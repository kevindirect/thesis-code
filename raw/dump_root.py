"""
Kevin Patel
"""
import sys
from os import sep
from os.path import basename
from functools import partial, reduce
import logging

import numpy as np
import pandas as pd

from common_util import RAW_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, DT_FMT_YMD_HM, load_json, load_df, benchmark, get_cmd_args, isnt, series_to_dti_noreindex, pd_before_cutoff, right_join, outer_join
from raw.common import default_joinsfile
from data.data_api import DataAPI
from data.data_util import make_entry


def dump_root(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	cmd_arg_list = ['joinsfile=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	joinsfile = default_joinsfile if (isnt(cmd_input['joinsfile='])) else cmd_input['joinsfile=']
	joins = load_json(joinsfile, dir_path=RAW_DIR)
	agg_freq = DT_CAL_DAILY_FREQ
	dti_freq = DT_HOURLY_FREQ
	dt_fmt = DT_FMT_YMD_HM

	for equity, file_list in joins.items():
		logging.info(equity)
		price = load_df(file_list['price'], dir_path=str(RAW_DIR +'price' +sep), \
			data_format='csv')
		vol = load_df(file_list['vol'], dir_path=str(RAW_DIR +'vol' +sep), \
			data_format='csv')
		joined = outer_join(price, vol)
		logging.info('joined market data...')

		for trmi_ver, trmi_ver_groups in file_list['trmi'].items():
			for trmi_cat, trmi_list in trmi_ver_groups.items():
				trmi_list_path = RAW_DIR +'trmi' +sep +trmi_ver +sep +trmi_cat +sep
				sents = [load_df(sent, dir_path=trmi_list_path, data_format='csv') \
					for sent in trmi_list]
				joined = fill_ver(reduce(outer_join, [joined] + sents))
		logging.info('joined all data...')

		# TODO - move index conversion to dti upstream to dump_price and dump_trmi
		joined.index = series_to_dti_noreindex(joined.index, fmt=dt_fmt, utc=True, exact=True, freq=dti_freq)
		joined = joined.asfreq(dti_freq)
		logging.info('converted index to dti with freq: {}...'.format(dti_freq))

		joined = joined.groupby(pd.Grouper(freq=agg_freq)).filter(pd_before_cutoff)
		logging.info('filtered days that don\'t begin at or before cutoff time...')

		logging.debug(joined.index, joined)
		DataAPI.dump(make_entry('raw', 'root', 'join', dti_freq, name=equity, cat=cat_map(file_list['price'])), joined)
		logging.info('dumped df')

def fill_ver(trmi_df):
	"""
	Front fills version columns (hacky fix)
	"""
	for col in trmi_df.columns:
		if (col.endswith('ver')):
			trmi_df[col] = trmi_df.loc[:, col].fillna(method='ffill').fillna(method='bfill')
	return trmi_df

def cat_map(primary_target):
	return {
		'DJI': 'us_equity_index',
		'SPX': 'us_equity_index',
		'NDX': 'us_equity_index',
		'RUT': 'us_equity_index',
		'USO': 'energy_material',
		'GLD': 'energy_material'
	}.get(primary_target, 'unspecified')


if __name__ == '__main__':
	with benchmark('ttf') as b:
		with DataAPI(async_writes=True):
			dump_root(sys.argv[1:])
