# Kevin Patel

import pandas as pd
import numpy as np
import datetime
import sys
import getopt
from os import getcwd, sep, path, makedirs


def main(argv):
	usage = lambda: print('clean.py -i <infile> -o <outfile> [-d -r]')
	pfx = getcwd() +sep
	frame = None
	outfname = None
	diff_sent = False
	round_sent = False

	try:
		opts, args = getopt.getopt(argv,'hi:o:dr',['help', 'infile=', 'outfile=', 'difference', 'round'])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt == ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-i', '--infile'):
			if (path.isfile(arg)):
				frame = pd.read_csv(arg, index_col=0)
			else:
				sys.exit(2)
		elif opt in ('-o', '--outfile'):
			outfname = arg
		elif opt in ('-d', '--difference'):
			diff_sent = True
		elif opt in ('-r', '--round'):
			round_sent = True

	if (frame is None):
		print('need to specify a dataframe to clean')
		usage()
		sys.exit(2)

	if (outfname is None):
		print('need to specify an output filename')
		usage()
		sys.exit(2)

	# Drop empty/null columns
	frame.drop('pba_volume', axis=1, inplace=True, errors='ignore')
	frame.drop('vol_numBids', axis=1, inplace=True, errors='ignore')
	frame.drop('vol_numAsks', axis=1, inplace=True, errors='ignore')

	# Frontfill mkt trmi data
	for col in frame.loc[:, frame.columns.str.startswith('mkt') & ~frame.columns.str.endswith('Ver')]:
		col = frame[col].fillna(method='ffill')
		if (diff_sent):
			col = col.diff()
		if (round_sent):
			col = col.round(5)
		frame[col] = col

	# Frontfill etf trmi data
	for col in frame.loc[:, frame.columns.str.startswith('etf') & ~frame.columns.str.endswith('Ver')]:
		frame[col] = frame[col].fillna(method='ffill').diff().round(5)

	# Make price label columns
	frame['ret_simple_oc'] = (frame['pba_close'] / frame['pba_open']) - 1
	frame['ret_simple_oa'] = (frame['pba_avgPrice'] / frame['pba_open']) - 1
	frame['ret_simple_oo'] = frame['pba_open'].pct_change()
	frame['ret_simple_cc'] = frame['pba_close'].pct_change()
	frame['ret_simple_aa'] = frame['pba_avgPrice'].pct_change()
	frame['ret_simple_hl'] = (frame['pba_high'] / frame['pba_low']) - 1
	frame['ret_dir_oc'] = np.sign(frame['ret_simple_oc'])
	frame['ret_dir_oa'] = np.sign(frame['ret_simple_oa'])
	frame['ret_dir_oo'] = np.sign(frame['ret_simple_oo'])
	frame['ret_dir_cc'] = np.sign(frame['ret_simple_cc'])
	frame['ret_dir_aa'] = np.sign(frame['ret_simple_aa'])

	# Make vol label columns
	frame['ret_vol_simple_oc'] = (frame['vol_close'] / frame['vol_open']) - 1
	frame['ret_vol_simple_oa'] = (frame['vol_avgPrice'] / frame['vol_open']) - 1
	frame['ret_vol_simple_oo'] = frame['vol_open'].pct_change()
	frame['ret_vol_simple_cc'] = frame['vol_close'].pct_change()
	frame['ret_vol_simple_aa'] = frame['vol_avgPrice'].pct_change()
	frame['ret_vol_simple_hl'] = (frame['vol_high'] / frame['vol_low']) - 1
	frame['ret_vol_dir_oc'] = np.sign(frame['ret_vol_simple_oc'])
	frame['ret_vol_dir_oa'] = np.sign(frame['ret_vol_simple_oa'])
	frame['ret_vol_dir_oo'] = np.sign(frame['ret_vol_simple_oo'])
	frame['ret_vol_dir_cc'] = np.sign(frame['ret_vol_simple_cc'])
	frame['ret_vol_dir_aa'] = np.sign(frame['ret_vol_simple_aa'])

	frame.to_csv(outfname)


if __name__ == '__main__':
	main(sys.argv[1:])
