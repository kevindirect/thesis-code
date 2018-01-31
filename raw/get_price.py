# Kevin Patel

import pandas as pd
import datetime
import sys
import getopt
import json
from os import getcwd, sep, path, makedirs, pardir
sys.path.insert(0, path.abspath(pardir))
from common import makedir_if_not_exists, clean_cols, month_num

def main(argv):
	usage = lambda: print('get_price.py [-f <filename> -p <pathsfile> -c <cleanfile>]')
	pfx = getcwd() +sep
	pricefile = 'richard@marketpsychdata.com--N166567660.csv'
	pathsfile = 'paths.json'
	cleanfile = 'clean.json'

	try:
		opts, args = getopt.getopt(argv,'hf:p:c:',['help', 'filename=', 'pathsfile=', 'cleanfile'])
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
		elif opt in ('-c', '--cleanfile'):
			cleanfile = arg

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

	# cleanfile contains basic processing for price and volatility data
	if (path.isfile(pfx +cleanfile)):
		with open(pfx +cleanfile) as json_data:
			clean_dict = json.load(json_data)
			price_clean = clean_dict['price']
			vol_clean = clean_dict['vol']
	else:
		print(cleanfile, 'must be present in the current directory')
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

		clean_instr = None
		if (name in price_clean):
			clean_instr = price_clean
		elif (name in vol_clean):
			clean_instr = vol_clean

		if (clean_instr is not None):
			print('\tcleaning ' +name, end='...', flush=True)
			asset_df = clean_cols(asset_df, clean_instr)
			asset_df = clean_cols(asset_df, clean_instr[name])
			print('done')

		if (name in price_path):
			asset_df.to_csv(pfx +'price' +sep +name +'.csv')
		elif (name in vol_path):
			asset_df.to_csv(pfx +'vol' +sep +name +'.csv')
		else:
			asset_df.to_csv(pfx +name +'.csv')

if __name__ == '__main__':
	main(sys.argv[1:])
