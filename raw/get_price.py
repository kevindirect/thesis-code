# Kevin Patel

import pandas as pd
import datetime
import sys
import getopt
import json
from os import getcwd, sep, path, makedirs, pardir
sys.path.insert(0, path.abspath(pardir))
from common import makedir_if_not_exists, month_num

def main(argv):
	usage = lambda: print('get_price.py [-f <filename> -p <pathsfile> -c <columnsfile> -r <rowsfile>]')
	pfx = getcwd() +sep
	pricefile = 'richard@marketpsychdata.com--N166567660.csv'
	pathsfile = 'paths.json'
	columnsfile = 'columns.json'
	rowsfile = 'rows.json'

	try:
		opts, args = getopt.getopt(argv,'hf:p:c:r:',['help', 'filename=', 'pathsfile=', 'columnsfile=', 'rowsfile='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-f', '--filename'):
			pricefile = arg
		elif opt in ('-p', '--pathsfile'):
			pathsfile = arg
		elif opt in ('-c', '--columnsfile'):
			columnsfile = arg
		elif opt in ('-r', '--rowsfile'):
			rowsfile = arg

	if (path.isfile(pfx +pricefile)):
		df = pd.read_csv(pfx +pricefile)
	else:
		print('default or alternate csv price file is required for this script to run')
		sys.exit(2)

	# pathsfile tells script what to pull from the price csv file and where to put it
	if (path.isfile(pfx +pathsfile)):
		with open(pfx +pathsfile) as json_data:
			path_dict = json.load(json_data)
			price_path = path_dict['price']
			vol_path = path_dict['vol']
	else:
		print(pathsfile, 'must be present in the current directory')
		sys.exit(2)

	# columnsfile contains processing directives for price and volatility dataframe columns
	if (path.isfile(pfx +columnsfile)):
		with open(pfx +columnsfile) as json_data:
			columns_dict = json.load(json_data)
			price_clean_cols = columns_dict['price']
			vol_clean_cols = columns_dict['vol']
	else:
		print(columnsfile, 'must be present in the current directory')
		sys.exit(2)

	# rowsfile contains processing directives for price and volatility dataframe rows
	if (path.isfile(pfx +rowsfile)):
		with open(pfx +rowsfile) as json_data:
			rows_dict = json.load(json_data)
			price_clean_rows = rows_dict['price']
			vol_clean_rows = rows_dict['vol']
	else:
		print(rowsfile, 'must be present in the current directory')
		sys.exit(2)

	makedir_if_not_exists(pfx +'price')	
	makedir_if_not_exists(pfx +'vol')

	for asset_code in df['#RIC'].unique():
		name = asset_code[-3:]
		print(name)
		asset_df = df.loc[df['#RIC'] == asset_code]

		# Convert to all numeric date and reverse to match TRMI format
		date_col = asset_df['Date[G]'].map(lambda w: (w[7:] + '-' + month_num[w[3:6]] + '-' + w[:2]))
		hour_col = asset_df['Time[G]'].map(lambda w: int(w[:2]))
		datehourmin_col = date_col.map(str) +' ' +hour_col.map(lambda h: str(h).zfill(2)) +':00'
		asset_df.insert(0, 'id', datehourmin_col)

		unchanged = [col for col in asset_df.columns if(len(asset_df[col].value_counts())<=1)]
		print('\tdropping', len(unchanged), 'unchanging columns', end='...', flush=True)
		asset_df = asset_df.drop(unchanged, axis=1, errors='ignore')
		print('done')
		asset_df.drop(['#RIC', 'Date[G]', 'Time[G]'], axis=1, errors='ignore', inplace=True)
		asset_df = asset_df.set_index('id', drop=True).sort_index()

		clean_cols_instr = None
		if (name in price_clean_cols):
			clean_cols_instr = price_clean_cols
		elif (name in vol_clean_cols):
			clean_cols_instr = vol_clean_cols

		clean_rows_instr = None
		if (name in price_clean_rows):
			clean_rows_instr = price_clean_rows
		elif (name in vol_clean_rows):
			clean_rows_instr = vol_clean_rows

		if (clean_cols_instr is not None):
			print('\tprocessing ' +name +' columns', end='...', flush=True)
			asset_df = clean_cols(asset_df, clean_cols_instr)
			asset_df = clean_cols(asset_df, clean_cols_instr[name])
			print('done')

		if (clean_rows_instr is not None):
			print('\tprocessing ' +name +' rows', end='...', flush=True)
			asset_df = clean_rows(asset_df, clean_rows_instr)
			asset_df = clean_rows(asset_df, clean_rows_instr[name])
			print('done')

		if (name in price_path):
			asset_df.to_csv(pfx +'price' +sep +name +'.csv')
		elif (name in vol_path):
			asset_df.to_csv(pfx +'vol' +sep +name +'.csv')
		else:
			asset_df.to_csv(pfx +name +'.csv')
		print()


def clean_cols(frame, clean_cols_instr):
	if ("drop" in clean_cols_instr):
		frame = frame.drop(clean_cols_instr["drop"], axis=1, errors='ignore')
	if ("rename" in clean_cols_instr):
		frame = frame.rename(columns=clean_cols_instr["rename"])
	if ("col_prefix" in clean_cols_instr):
		frame.columns = frame.columns.map(lambda s: str(clean_cols_instr["col_prefix"] +s))
	return frame

def clean_rows(frame, clean_rows_instr):
	if ("filter" in clean_rows_instr):
		row_filter = clean_rows_instr["filter"]
		if ("null" in row_filter):
			for col_set in row_filter["null"]:
				frame = frame.dropna(axis=0, how='all', subset=col_set)

		if ("zero" in row_filter):
			for col_set in row_filter["zero"]:
				frame = frame[(frame[col_set] != 0).all(axis=1)]

		if ("one" in row_filter):
			for col_set in row_filter["one"]:
				frame = frame[(frame[col_set] != 1).all(axis=1)]

		if ("same" in row_filter):
			for col_set in row_filter["same"]:
				assert(len(col_set)>1)
				frame = frame[~frame[col_set].eq(frame[col_set[0]], axis=0).all(axis=1)]
	return frame

if __name__ == '__main__':
	main(sys.argv[1:])
