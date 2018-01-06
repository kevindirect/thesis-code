# Kevin Patel

import pandas as pd
import datetime
import sys
import getopt
from os import getcwd, sep, path, makedirs
sys.path.insert(0, path.abspath('..'))
from common import *

def main(argv):
	usage = lambda: print('split_price.py -f <filename>')
	pfx = getcwd() +sep

	try:
		opts, args = getopt.getopt(argv,'hf:',['help', 'filename'])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	price_file = None
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-f', '--filename'):
			price_file = arg

	if not price_file:
		print('price csv file is a required commandline argument')
		sys.exit(2)

	price = ['DJI', 'SPX', 'NDX', 'RUT', 'USO', 'GLD']
	vol = ['VXD', 'VIX', 'VXN', 'RVX', 'OVX', 'GVZ']
	mon_str_to_num = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
					'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}

	makedir_if_not_exists(pfx +'price')	
	makedir_if_not_exists(pfx +'vol')

	df = pd.read_csv(price_file)

	for asset_code in df['#RIC'].unique():
		name = asset_code[-3:]
		print(name, end='')
		asset_df = df.loc[df['#RIC'] == asset_code]

		# Convert to all numeric date and reverse to match TRMI format
		date_col = asset_df['Date[G]'].map(lambda w: (w[7:] + '-' + mon_str_to_num[w[3:6]] + '-' + w[:2]))
		hour_col = asset_df['Time[G]'].map(lambda w: int(w[:2]))
		datehourmin_col = date_col.map(str) +' ' +hour_col.map(lambda h: str(h).zfill(2)) +':00'
		dow_col = date_col.map(lambda w: datetime.datetime.strptime(w, '%Y-%m-%d').weekday())
		day_col = date_col.map(lambda w: int(w[8:10]))
		month_col = date_col.map(lambda w: int(w[5:7]))
		year_col = date_col.map(lambda w: int(w[:4]))

		asset_df.insert(0, 'id', datehourmin_col)
		asset_df.insert(1, 'date', date_col)
		asset_df.insert(2, 'hour', hour_col)
		asset_df.insert(3, 'gmt_offset', asset_df['GMT Offset'])
		asset_df.insert(4, 'dow', dow_col)
		asset_df.insert(5, 'day', day_col)
		asset_df.insert(6, 'month', month_col)
		asset_df.insert(7, 'year', year_col)

		asset_df = asset_df.drop(['#RIC', 'Date[G]', 'Time[G]', 'GMT Offset'], axis=1, errors='ignore') #Drop redundant date/time columns
		asset_df = asset_df.set_index('id', drop=True).sort_index()

		if (name in price):
			asset_df.to_csv(pfx +'price' +sep +name +'.csv')
		elif (name in vol):
			asset_df.to_csv(pfx +'vol' +sep +name +'.csv')
		else:
			asset_df.to_csv(pfx +'price' +sep +name +'.csv')
		print('...done')

if __name__ == '__main__':
	main(sys.argv[1:])
