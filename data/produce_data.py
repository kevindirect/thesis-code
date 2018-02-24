# Kevin Patel

import pandas as pd
import numpy as np
import sys
import getopt
from os import getcwd, sep, path, makedirs, pardir
import json
from functools import reduce
sys.path.insert(0, path.abspath(pardir))
from common_util import makedir_if_not_exists, right_join, inner_join, outer_join, load_csv, get_subset
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
		price = load_csv(raw_data_dir +'price' +sep +file_list['price'] +'.csv')
		vol = load_csv(raw_data_dir +'vol' +sep +file_list['vol'] +'.csv')
		joined = outer_join(price, vol)

		for trmi_ver, trmi_ver_groups in file_list['trmi'].items():
			for trmi_cat, trmi_list in trmi_ver_groups.items():
				trmi_list_path = raw_data_dir +'trmi' +sep +trmi_ver +sep +trmi_cat +sep
				sents = [load_csv(trmi_list_path +sent +'.csv') for sent in trmi_list]
				joined = reduce(right_join, [joined] + sents)

		joined = inner_join(joined, make_time_cols(joined))
		joined = inner_join(joined, make_label_cols(joined))

		for split_group, split_list in splits.items():
			print('\tsplit', split_group)
			equity_dir = pfx +split_group +sep +equity +sep
			makedir_if_not_exists(equity_dir)

			for split, qualifiers in split_list['#ASSET'].items():
				print('\t\t' +split, end='...', flush=True)
				split_columns = get_subset(joined.columns, qualifiers)
				joined[split_columns].dropna(axis=0, how='all').to_csv(equity_dir +split +'.csv')
				print('done')
		print()

if __name__ == '__main__':
	main(sys.argv[1:])
