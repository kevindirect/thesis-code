# Kevin Patel

import sys
import getopt
from os import sep
from functools import reduce
from common import load_json, load_csv, makedir_if_not_exists, right_join, inner_join, outer_join, get_subset
from common import RAW_DIR, DATA_DIR, default_joinfile, default_splitfile
from add_columns import make_time_cols, make_label_cols


def produce_data(argv):
	usage = lambda: print('produce_data.py [-j <joinfile> -s <splitfile>]')
	joinfile = default_joinfile
	splitfile = default_splitfile

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
	joins = load_json(joinfile, dir_path=DATA_DIR)

	# splitfile tells script what columns to split into separate frames
	splits = load_json(splitfile, dir_path=DATA_DIR)

	for equity, file_list in joins.items():
		print('processing', equity)
		price = load_csv(str(file_list['price'] +'.csv'), dir_path=str(RAW_DIR +'price' +sep))
		vol = load_csv(str(file_list['vol'] +'.csv'), dir_path=str(RAW_DIR +'vol' +sep))
		joined = outer_join(price, vol)

		for trmi_ver, trmi_ver_groups in file_list['trmi'].items():
			for trmi_cat, trmi_list in trmi_ver_groups.items():
				trmi_list_path = RAW_DIR +'trmi' +sep +trmi_ver +sep +trmi_cat +sep
				sents = [load_csv(str(sent +'.csv'), dir_path=trmi_list_path) for sent in trmi_list]
				joined = reduce(right_join, [joined] + sents)

		joined = inner_join(joined, make_time_cols(joined))
		joined = inner_join(joined, make_label_cols(joined))

		for split_group, split_list in splits.items():
			print('\tsplit', split_group)
			equity_dir = DATA_DIR +split_group +sep +equity +sep
			makedir_if_not_exists(equity_dir)

			for split, qualifiers in split_list['#ASSET'].items():
				print('\t\t' +split, end='...', flush=True)
				split_columns = get_subset(joined.columns, qualifiers)
				joined[split_columns].dropna(axis=0, how='all').to_csv(equity_dir +split +'.csv')
				print('done')
		print()

if __name__ == '__main__':
	produce_data(sys.argv[1:])
