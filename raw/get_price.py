"""
Kevin Patel
"""
import sys
from os import sep
from os.path import basename
import logging

from common_util import RAW_DIR, load_json, get_cmd_args, isnt, makedir_if_not_exists, dump_df
from raw.common import default_pricefile, default_pathsfile, default_columnsfile, default_rowsfile, load_csv_no_idx


def get_price(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	cmd_args_list = ['filename=', 'pathsfile=', 'columnsfile=', 'rowsfile=']
	cmd_args = get_cmd_args(argv, cmd_args_listi, script_name=basename(__file__))
	pricefile = default_pricefile if (isnt(cmd_args['filename='])) else cmd_args['filename=']
	pathsfile = default_pathsfile if (isnt(cmd_args['pathsfile='])) else cmd_args['pathsfile=']
	columnsfile = default_columnsfile if (isnt(cmd_args['columnsfile='])) else cmd_args['columnsfile=']
	rowsfile = default_rowsfile if (isnt(cmd_args['rowsfile='])) else cmd_args['rowsfile=']

	MONTH_NUM = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}

	# load pricefile into a dataframe with a generated index
	df = load_csv_no_idx(pricefile, dir_path=RAW_DIR)

	# pathsfile tells script what to pull from the price csv file and where to put it
	path_dict = load_json(pathsfile, dir_path=RAW_DIR)
	price_path = path_dict['price']
	vol_path = path_dict['vol']

	# columnsfile contains processing directives for price and volatility dataframe columns
	columns_dict = load_json(columnsfile, dir_path=RAW_DIR)
	price_clean_cols = columns_dict['price']
	vol_clean_cols = columns_dict['vol']

	# rowsfile contains processing directives for price and volatility dataframe rows
	rows_dict = load_json(rowsfile, dir_path=RAW_DIR)
	price_clean_rows = rows_dict['price']
	vol_clean_rows = rows_dict['vol']

	makedir_if_not_exists(RAW_DIR +'price')
	makedir_if_not_exists(RAW_DIR +'vol')

	for asset_code in df['#RIC'].unique():
		asset = asset_code[-3:]
		print(asset)
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
		if (asset in price_clean_cols):
			clean_cols_instr = price_clean_cols
		elif (asset in vol_clean_cols):
			clean_cols_instr = vol_clean_cols

		clean_rows_instr = None
		if (asset in price_clean_rows):
			clean_rows_instr = price_clean_rows
		elif (asset in vol_clean_rows):
			clean_rows_instr = vol_clean_rows

		if (clean_cols_instr is not None):
			print('\tprocessing ' +asset +' columns', end='...', flush=True)
			asset_df = clean_cols(asset_df, clean_cols_instr)
			asset_df = clean_cols(asset_df, clean_cols_instr[asset])
			print('done')

		if (clean_rows_instr is not None):
			print('\tprocessing ' +asset +' rows', end='...', flush=True)
			asset_df = clean_rows(asset_df, clean_rows_instr)
			asset_df = clean_rows(asset_df, clean_rows_instr[asset])
			print('done')

		target_dir = RAW_DIR
		target_dir += 'price' +sep if asset in price_path else ''
		target_dir += 'vol' +sep if asset in vol_path else ''
		dump_df(asset_df, asset, dir_path=target_dir)
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
