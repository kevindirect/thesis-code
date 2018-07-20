# Kevin Patel

import sys
import getopt
from os import sep
from functools import reduce

from common_util import RAW_DIR, load_json, load_df, series_to_dti, right_join, outer_join
from raw.common import default_joinsfile
from data.data_api import DataAPI


def dump_raw(argv):
	usage = lambda: print('dump_raw.py [-j <joinsfile>]')
	joinsfile = default_joinsfile

	try:
		opts, args = getopt.getopt(argv, 'hj:', ['help', 'joinsfile='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-j', '--joinsfile'):
			joinsfile = arg

	# joinsfile tells script what files to join together
	joins = load_json(joinsfile, dir_path=RAW_DIR)

	for equity, file_list in joins.items():
		print(equity)
		print('\tjoining', end='...', flush=True)
		price = load_df(file_list['price'], dir_path=str(RAW_DIR +'price' +sep))
		vol = load_df(file_list['vol'], dir_path=str(RAW_DIR +'vol' +sep))
		joined = outer_join(price, vol)

		for trmi_ver, trmi_ver_groups in file_list['trmi'].items():
			for trmi_cat, trmi_list in trmi_ver_groups.items():
				trmi_list_path = RAW_DIR +'trmi' +sep +trmi_ver +sep +trmi_cat +sep
				sents = [load_df(sent, dir_path=trmi_list_path) for sent in trmi_list]
				joined = reduce(right_join, [joined] + sents)
		print('done')

		entry = {
			'root': equity,
			'basis': equity,
			'stage': 'raw',
			'desc': 'raw'
			'raw_cat': cat_map(file_list['price'])
		}
		print('\tdumping', end='...', flush=True)
		joined.index = series_to_dti(joined.index) # XXX - move index conversion to dti upstream
		DataAPI.dump(joined, entry)
		print('done')

	DataAPI.update_record()


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
	dump_raw(sys.argv[1:])
