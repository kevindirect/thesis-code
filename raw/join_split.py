# Kevin Patel

import pandas as pd
import numpy as np
import datetime
import sys
import getopt
from os import getcwd, sep, path, makedirs, pardir
import json
sys.path.insert(0, path.abspath(pardir))
from common import makedir_if_not_exists

def main(argv):
	usage = lambda: print('join_split.py [-p <pathsfile> -j <joinfile>]')
	pfx = getcwd() +sep
	pathsfile = 'paths.json'
	joinfile = 'join.json'

	try:
		opts, args = getopt.getopt(argv,'hp:j:',['help', 'pathsfile=', 'joinfile='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-p', '--pathsfile'):
			pathsfile = arg
		elif opt in ('-j', '--joinfile'):
			joinfile = arg

	# pathsfile tells script where everything is
	# if (path.isfile(pfx +pathsfile)):
	# 	with open(pfx +pathsfile) as json_data:
	# 		trmi_paths = json.load(json_data)['trmi']
	# else:
	# 	print(pathsfile, 'must be present in the current directory')
	# 	sys.exit(2)

	# joinfile tells script what files to join together
	if (path.isfile(pfx +joinfile)):
		with open(pfx +joinfile) as json_data:
			join_paths = json.load(json_data)['trmi']
	else:
		print(joinfile, 'must be present in the current directory')
		sys.exit(2)

	# join_cols = ['id', 'date', 'hour', 'day', 'month', 'year']
	# price_vol_renames = {'Last': 'Close', 'Ave. Price': 'Avg. Price',
	# 	'No. Trades': 'Num. Trades', 'No. Bids': 'Num. Bids', 'No. Asks': 'Num. Asks'}
	# pba = ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg. Price', 'Num. Trades',
	#     'Open Bid', 'High Bid', 'Low Bid', 'Close Bid', 'Num. Bids',
	#     'Open Ask', 'High Ask', 'Low Ask', 'Close Ask', 'Num. Asks']
	# uncap = lambda s: s[:1].lower() + s[1:] if s else ''
	# conv = lambda s: uncap(str(s).title()).replace('.', '').replace(' ', '')
	# makedir_if_not_exists(pfx +'dirty')

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

		# dow_col = date_col.map(lambda w: datetime.datetime.strptime(w, '%Y-%m-%d').weekday())
		# day_col = date_col.map(lambda w: int(w[8:10]))
		# month_col = date_col.map(lambda w: int(w[5:7]))
		# year_col = date_col.map(lambda w: int(w[:4]))

		asset_df.insert(0, 'id', datehourmin_col)
		# asset_df.insert(1, 'date', date_col)
		# asset_df.insert(2, 'hour', hour_col)
		# asset_df.insert(3, 'gmt_offset', asset_df['GMT Offset'])
		# asset_df.insert(4, 'dow', dow_col)
		# asset_df.insert(5, 'day', day_col)
		# asset_df.insert(6, 'month', month_col)
		# asset_df.insert(7, 'year', year_col)

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

				# asset_df.insert(1, 'date', date_col)
				# asset_df.insert(2, 'hour', asset_df['windowTimestamp'].map(lambda w: int(w[11:13])))
				# asset_df.insert(3, 'dow', date_col.map(lambda w: datetime.datetime.strptime(w, '%Y-%m-%d').weekday()))
				# asset_df.insert(4, 'day', asset_df['windowTimestamp'].map(lambda w: int(w[8:10])))
				# asset_df.insert(5, 'month', asset_df['windowTimestamp'].map(lambda w: int(w[5:7])))
				# asset_df.insert(6, 'year', asset_df['windowTimestamp'].map(lambda w: int(w[:4])))