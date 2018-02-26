# Kevin Patel

import sys
import getopt
from os import sep
from common import load_json, load_csv, makedir_if_not_exists, MONTH_NUM
from common import RAW_DIR, default_pricefile, default_pathsfile, default_columnsfile, default_rowsfile


def get_price(argv):
	usage = lambda: print('get_price.py [-f <filename> -p <pathsfile> -c <columnsfile> -r <rowsfile>]')
	pricefile = default_pricefile
	pathsfile = default_pathsfile
	columnsfile = default_columnsfile
	rowsfile = default_rowsfile

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

	# load pricefile into a dataframe with a generated index
	df = load_csv(pricefile, idx_0=False)

	# pathsfile tells script what to pull from the price csv file and where to put it
	path_dict = load_json(pathsfile)
	price_path = path_dict['price']
	vol_path = path_dict['vol']

	# columnsfile contains processing directives for price and volatility dataframe columns
	columns_dict = load_json(columnsfile)
	price_clean_cols = columns_dict['price']
	vol_clean_cols = columns_dict['vol']

	# rowsfile contains processing directives for price and volatility dataframe rows
	rows_dict = load_json(rowsfile)
	price_clean_rows = rows_dict['price']
	vol_clean_rows = rows_dict['vol']

	makedir_if_not_exists(RAW_DIR +'price')	
	makedir_if_not_exists(RAW_DIR +'vol')

	for asset_code in df['#RIC'].unique():
		name = asset_code[-3:]
		print(name)
		asset_df = df.loc[df['#RIC'] == asset_code]

		# Convert to all numeric date and reverse to match TRMI format
		date_col = asset_df['Date[G]'].map(lambda w: (w[7:] + '-' + MONTH_NUM[w[3:6]] + '-' + w[:2]))
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
			asset_df.to_csv(RAW_DIR +'price' +sep +name +'.csv')
		elif (name in vol_path):
			asset_df.to_csv(RAW_DIR +'vol' +sep +name +'.csv')
		else:
			asset_df.to_csv(RAW_DIR +name +'.csv')
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
	get_price(sys.argv[1:])
