# Kevin Patel

import pandas as pd
import numpy as np
import re
import sys
import getopt
from os import getcwd, sep, path, makedirs, pardir
import json
sys.path.insert(0, path.abspath(pardir))
from common import makedir_if_not_exists
from filter_rows import *
from add_columns import make_time_cols, make_label_cols


def main(argv):
	usage = lambda: print('produce_data.py [-j <joinfile> -s <splitfile>]')
	pfx = getcwd() +sep
	raw_data_dir = path.abspath(pardir) +sep +'raw' +sep
	joinfile = 'join.json'
	splitfile = 'split.json'

	try:
		opts, args = getopt.getopt(argv, 'hj:s:', ['help', 'joinfile=', 'splitfile='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-j', '--joinfile'):
			joinfile = arg
		elif opt in ('-s', '--splitfile'):
			splitfile = arg

	# joinfile tells script what files to join together
	if (path.isfile(pfx +joinfile)):
		with open(pfx +joinfile) as json_data:
			joins = json.load(json_data)
	else:
		print(joinfile, 'must be present in the current directory')
		sys.exit(2)

	# splitfile tells script what columns to split into separate frames
	if (path.isfile(pfx +splitfile)):
		with open(pfx +splitfile) as json_data:
			splits = json.load(json_data)
	else:
		print(splitfile, 'must be present in the current directory')
		sys.exit(2)

	for equity, file_list in joins.items():
		print('processing', equity)
		price = pd.read_csv(raw_data_dir +'price' +sep +file_list['price'] +'.csv', index_col=0)
		vol = pd.read_csv(raw_data_dir +'vol' +sep +file_list['vol'] +'.csv', index_col=0)
		joined = price.merge(vol, how='outer', left_index=True, right_index=True, sort=True)

		for trmi_ver, trmi_ver_groups in file_list['trmi'].items():
			for trmi_cat, trmi_list in trmi_ver_groups.items():
				trmi_list_path = raw_data_dir +'trmi' +sep +trmi_ver +sep +trmi_cat +sep
				for trmi_sent in trmi_list:
					sent = pd.read_csv(trmi_list_path +trmi_sent +'.csv', index_col=0)
					joined = joined.merge(sent, how='outer', left_index=True, right_index=True, sort=True)

		# TODO - Row cleaning here

		joined = joined.merge(make_time_cols(joined), how='outer', left_index=True, right_index=True, sort=True)
		joined = joined.merge(make_label_cols(joined), how='outer', left_index=True, right_index=True, sort=True)

		for split_group, split_list in splits.items():
			equity_dir = pfx +split_group +sep +equity +sep
			makedir_if_not_exists(equity_dir)

			for split, qualifier in split_list.items():
				print('\t' +split, end='...', flush=True)
				split_columns = []
				split_columns.extend(qualifier['exact'])
				split_columns.extend([col for col in joined.columns if col.startswith(tuple(qualifier['startswith']))])
				split_columns.extend([col for col in joined.columns if col.endswith(tuple(qualifier['endswith']))])
				split_columns.extend([col for col in joined.columns if (any(re.match(rgx, col) for rgx in qualifier['regex']))])
				if (qualifier['exclude']):
					split_columns = [col for col in split_columns if col not in qualifier['exclude']]
				joined[split_columns].to_csv(equity_dir +split +'.csv')
				print('done')
		print()
				
if __name__ == '__main__':
	main(sys.argv[1:])
