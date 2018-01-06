# Kevin Patel
# TRMI api ver 1.1

import pandas as pd
import datetime
import sys
import getopt
from os import getcwd, sep, path, makedirs
sys.path.insert(0, path.abspath('..'))
from common import *

def main(argv):
	usage = lambda: print('get_trmi.py -v <version> [-k <keep_ns>]')
	ver = None
	valid_ver = ['v2', 'v3']

	# Default Parameters / login info
	url = 'https://api.marketpsych.com'
	key = 'cus_BgEA801LPsAccR'
	per = 'hourly'   # daily, hourly, or minutely
	keep_ns = False  # Set True to keep News_Social data in result
	startend = {'start': '1998-01-01', 'end':'2017-12-23'}

	try:
		opts, args = getopt.getopt(argv,'hv:k:', ['help', 'version=', 'keep_ns='])
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
		elif opt in ('-k', '--keep_ns'):
			if (arg == 'True'):
				keep_ns = True
			elif (arg == 'False'):
				keep_ns = False
			else:
				print('keep_ns must be a True or False string')
				sys.exit(2)

	if (not ver):
		print('need to specify a version')
		usage()
		sys.exit(2)

	asset_lists_v2 = {
		'etf': ['MPTRXUS30', 'MPTRXUS500', 'MPTRXUSMID2000', 'MPTRXUSNAS100'], # CMPNY_GRP
		'enm' : ['CRU', 'GOL'] # COM_ENM
	}
	asset_lists_v3 = {
		'coumkt': ['US'], # COU_MKT
		'etf': ['MPTRXUS30', 'MPTRXUS500', 'MPTRXUSMID2000', 'MPTRXUSNAS100'], # CMPNY_GRP
		'enm' : ['CRU', 'GOL'] # COM_ENM
	}
	asset_lists = asset_lists_v2 if ver == 'v2' else asset_lists_v3

	def make_csv_group_request_url(group, assets, period=per, times=startend, version=ver, start_url=url, apikey=key):
		reqlist = []
		params = []

		params.append(period +'?apikey=' +apikey)
		params.append('assets=' +','.join(assets))
		params.append('start=' +times['start'])
		params.append('end=' +times['end'])
		params.append('csv=1')

		reqlist.append(start_url)
		reqlist.append('trmi')
		reqlist.append(version)
		reqlist.append('data')
		reqlist.append(group)
		reqlist.append('&'.join(params))

		return '/'.join(reqlist)

	dropfirst = ['id', 'Date', 'Asset']
	join_cols = ['assetCode', 'windowTimestamp']

	for group, assets in asset_lists.items():
		endpoint = make_csv_group_request_url(group, assets)
		print(group, endpoint)
		df = pd.read_csv(endpoint)
		dir_path = getcwd() +sep +'trmi' +ver +sep +group
		makedir_if_not_exists(dir_path)
		# with open(dir_path +sep +'metadata.txt', 'a') as text_file:
		# 	text_file.write('asset: ' +str(df['Asset'].unique()) +'\n')
		# 	text_file.write('group: ' +group +'\n')
		# 	text_file.write('version: ' +str(df['systemVersion'].unique()) +'\n')
		# 	text_file.write('\n')

		df = df.drop(dropfirst, axis=1, errors='ignore')
		news = df.loc[df['dataType'] == 'News'].drop('dataType', axis=1)
		social = df.loc[df['dataType'] == 'Social'].drop('dataType', axis=1)
		merged = news.merge(social, on=join_cols, suffixes=('_N', '_S'))
		if (keep_ns):
			news_social = df.loc[df['dataType'] == 'News_Social'].drop('dataType', axis=1)
			merged = merged.merge(news_social, on=join_cols, suffixes=('', '_NS'))

		for asset in assets:
			print('\t' +asset, end='')
			asset_df = merged.loc[merged['assetCode'] == asset]

			datehourmin_col = asset_df['windowTimestamp'].map(lambda w: w[:16])
			date_col = asset_df['windowTimestamp'].map(lambda w: w[:10])
			hour_col = asset_df['windowTimestamp'].map(lambda w: int(w[11:13]))
			dow_col = date_col.map(lambda w: datetime.datetime.strptime(w, '%Y-%m-%d').weekday())
			day_col = asset_df['windowTimestamp'].map(lambda w: int(w[8:10]))
			month_col = asset_df['windowTimestamp'].map(lambda w: int(w[5:7]))
			year_col = asset_df['windowTimestamp'].map(lambda w: int(w[:4]))

			asset_df.insert(0, 'id', datehourmin_col)
			asset_df.insert(1, 'date', date_col)
			asset_df.insert(2, 'hour', hour_col)
			asset_df.insert(3, 'dow', dow_col)
			asset_df.insert(4, 'day', day_col)
			asset_df.insert(5, 'month', month_col)
			asset_df.insert(6, 'year', year_col)
			asset_df.insert(7, 'news_ver', asset_df['systemVersion_N'])
			asset_df.insert(8, 'social_ver', asset_df['systemVersion_S'])
			asset_df = asset_df.drop(['systemVersion_N', 'systemVersion_S'], axis=1, errors='ignore')
			if (keep_ns):
				asset_df.insert(9, 'news_social_ver', asset_df['systemVersion_NS'])
				asset_df = asset_df.drop('systemVersion_NS', axis=1, errors='ignore')

			asset_df = asset_df.drop(join_cols, axis=1)
			asset_df = asset_df.set_index('id', drop=True).sort_index()
			asset_df.to_csv(dir_path +sep +asset +'.csv')
			print('...done')
		print()

if __name__ == '__main__':
	main(sys.argv[1:])
