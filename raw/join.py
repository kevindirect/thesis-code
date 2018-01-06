# Kevin Patel

import pandas as pd
import numpy as np
import datetime
import sys
import getopt
from os import getcwd, sep, path, makedirs
sys.path.insert(0, path.abspath('..'))
from common import *

def main(argv):
	usage = lambda: print('join.py -v <version>')
	ver = None
	valid_ver = ['v2', 'v3']
	pfx = getcwd() +sep

	try:
		opts, args = getopt.getopt(argv,'hv:',['help', 'version='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-v', '--version'):
			ver = arg
			if ver not in valid_ver:
				print('invalid version - the following are valid versions:', ', '.join(valid_ver))
				sys.exit(2)

	if (not ver):
		print('need to specify a version')
		usage()
		sys.exit(2)

	frames_v2 = {
		'DJI': [('vol', 'VXD'), ('trmiv2' +sep +'etf', 'MPTRXUS30')],
		'SPX': [('vol', 'VIX'), ('trmiv2' +sep +'etf', 'MPTRXUS500')],
		'NDX': [('vol', 'VXN'), ('trmiv2' +sep +'etf', 'MPTRXUSNAS100')],
		'RUT': [('vol', 'RVX'), ('trmiv2' +sep +'etf', 'MPTRXUSMID2000')],
		'USO': [('vol', 'OVX'), ('trmiv2' +sep +'enm', 'CRU')],
		'GLD': [('vol', 'GVZ'), ('trmiv2' +sep +'enm', 'GOL')]
	}
	frames_v3 = {
		'DJI': [('vol', 'VXD'), ('trmiv3' +sep +'coumkt', 'US'), ('trmiv3' +sep +'etf', 'MPTRXUS30')],
		'SPX': [('vol', 'VIX'), ('trmiv3' +sep +'coumkt', 'US'), ('trmiv3' +sep +'etf', 'MPTRXUS500')],
		'NDX': [('vol', 'VXN'), ('trmiv3' +sep +'coumkt', 'US'), ('trmiv3' +sep +'etf', 'MPTRXUSNAS100')],
		'RUT': [('vol', 'RVX'), ('trmiv3' +sep +'coumkt', 'US'), ('trmiv3' +sep +'etf', 'MPTRXUSMID2000')],
		'USO': [('vol', 'OVX'), ('trmiv3' +sep +'enm', 'CRU')],
		'GLD': [('vol', 'GVZ'), ('trmiv3' +sep +'enm', 'GOL')]
	}
	frames = frames_v2 if ver == 'v2' else frames_v3

	join_cols = ['id', 'date', 'hour', 'day', 'month', 'year']
	price_vol_renames = {'Last': 'Close', 'Ave. Price': 'Avg. Price',
		'No. Trades': 'Num. Trades', 'No. Bids': 'Num. Bids', 'No. Asks': 'Num. Asks'}
	pba = ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg. Price', 'Num. Trades',
	    'Open Bid', 'High Bid', 'Low Bid', 'Close Bid', 'Num. Bids',
	    'Open Ask', 'High Ask', 'Low Ask', 'Close Ask', 'Num. Asks']
	uncap = lambda s: s[:1].lower() + s[1:] if s else ''
	conv = lambda s: uncap(str(s).title()).replace('.', '').replace(' ', '')
	makedir_if_not_exists(pfx +'dirty')

	for price, data_list in frames.items():
		print(price)
		join_df = pd.read_csv(pfx +'price' +sep +price +'.csv')
		join_df.rename(index=str, columns=price_vol_renames, inplace=True)
		join_df.columns = join_df.columns.map(lambda s: str('pba_' +conv(s)) if s in pba else str(s))

		for other_data in data_list:
			col_pfx = other_data[0][-3:]
			print('\t' +col_pfx, end='...')
			data_df = pd.read_csv(pfx +other_data[0] +sep +other_data[1] +'.csv')
			data_df = data_df.drop('dow', axis=1, errors='ignore')
			conv_except = {}
			if (col_pfx == 'vol'):
				conv_except['gmt_offset'] = str(col_pfx +'_gmt_offset')
				data_df.rename(index=str, columns=price_vol_renames, inplace=True)
				data_df.rename(index=str, columns=conv_except, inplace=True)
			data_df.columns = data_df.columns.map(lambda s: str(col_pfx +'_' +conv(s))
				if s not in join_cols+list(conv_except.values()) else str(s))
			join_df = pd.merge(join_df, data_df, on=join_cols, how='outer')
			print('done')

		# Update day of week column to account for nonmarket hours being added via trmi outer joins
		dow_col = join_df['date'].map(lambda w: datetime.datetime.strptime(w, '%Y-%m-%d').weekday())
		join_df['dow'] = dow_col

		# Drop unchanging or empty columns (one or fewer unique non-null values)
		unchanged = [col for col in join_df.columns if(len(join_df[col].value_counts())<=1)]
		print('\tdropping', len(unchanged), 'unchanging columns', end='...')
		join_df.drop(unchanged, axis=1, inplace=True)
		print('done')

		join_df = join_df.set_index('id', drop=True).sort_index()
		join_df.to_csv(pfx +'dirty' +sep +price +'_' +ver +'.csv')

if __name__ == '__main__':
	main(sys.argv[1:])
