# Kevin Patel
# TRMI api ver 1.1

import pandas as pd
import datetime
import sys
import getopt
from os import getcwd, sep, path, makedirs, pardir
import json
sys.path.insert(0, path.abspath(pardir))
from common import makedir_if_not_exists

def main(argv):
	usage = lambda: print('get_trmi.py [-p <pathsfile> -k -t]')
	pfx = getcwd() +sep
	trmi_info_dir = path.abspath(pardir) +sep

	# trmi.json file contains api url and key
	if (path.isfile(trmi_info_dir +'trmi.json')):
		with open(trmi_info_dir +'trmi.json') as json_data:
			trmi = json.load(json_data)
	else:
		print('trmi.json must be present in the following directory:', trmi_info_dir)
		sys.exit(2)

	# Default Parameters
	per = 'hourly'   			# daily, hourly, or minutely
	pathsfile = 'paths.json'
	keep_ns = False
	startend = {'start': '1998-01-01', 'end':'2018-01-07'}

	try:
		opts, args = getopt.getopt(argv,'hp:kt', ['help', 'pathsfile=', 'keep_ns', 'test'])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-p', '--pathsfile'):
			pathsfile = arg
		elif opt in ('-k', '--keep_ns'):
			keep_ns = True
		elif opt in ('-t', '--test'):
			startend = {'start': '2015-08-01', 'end':'2015-08-03'}

	# paths.json tells script what to pull from the api and where to put it
	if (path.isfile(pfx +pathsfile)):
		with open(pfx +pathsfile) as json_data:
			trmi_paths = json.load(json_data)['trmi']
	else:
		print(pathsfile, ' must be present in the current directory')
		sys.exit(2)

	dropfirst = ['id', 'Date', 'Asset']
	join_cols = ['assetCode', 'windowTimestamp']
	for ver, groups in trmi_paths.items():
		for group, assets in groups.items():
			endpoint = make_csv_group_request_url(group, assets, ver, per, startend, trmi['api']['url'], trmi['api']['key'])
			print(group, endpoint)
			df = pd.read_csv(endpoint)
			dir_path = pfx +'trmi' +sep +ver +sep +group
			makedir_if_not_exists(dir_path)

			df = df.drop(dropfirst, axis=1, errors='ignore')
			news = df.loc[df['dataType'] == 'News'].drop('dataType', axis=1)
			social = df.loc[df['dataType'] == 'Social'].drop('dataType', axis=1)
			merged = news.merge(social, on=join_cols, suffixes=('_N', '_S'))
			if (keep_ns):
				news_social = df.loc[df['dataType'] == 'News_Social'].drop('dataType', axis=1)
				merged = merged.merge(news_social, on=join_cols, suffixes=('', '_NS'))

			# Rename systemVersion and prefix all data columns with group name
			merged.rename(columns={'systemVersion_N': 'n_ver', 'systemVersion_S': 's_ver', 'systemVersion': 'ver'}, inplace=True)
			merged.columns = merged.columns.map(lambda s: str(group[-3:] +'_' +s) if s not in join_cols else s)

			for asset in assets:
				print('\t' +asset, end='')
				asset_df = merged.loc[merged['assetCode'] == asset]

				date_col = asset_df['windowTimestamp'].map(lambda w: w[:10])
				asset_df.insert(0, 'id', asset_df['windowTimestamp'].map(lambda w: w[:16]))
				asset_df.insert(1, 'date', date_col)
				asset_df.insert(2, 'hour', asset_df['windowTimestamp'].map(lambda w: int(w[11:13])))
				asset_df.insert(3, 'dow', date_col.map(lambda w: datetime.datetime.strptime(w, '%Y-%m-%d').weekday()))
				asset_df.insert(4, 'day', asset_df['windowTimestamp'].map(lambda w: int(w[8:10])))
				asset_df.insert(5, 'month', asset_df['windowTimestamp'].map(lambda w: int(w[5:7])))
				asset_df.insert(6, 'year', asset_df['windowTimestamp'].map(lambda w: int(w[:4])))

				asset_df = asset_df.drop(join_cols, axis=1)
				asset_df = asset_df.set_index('id', drop=True).sort_index()
				asset_df.to_csv(dir_path +sep +asset +'.csv')
				print('...done')
			print()

def make_csv_group_request_url(group, assets, version, period, times, api_url, api_key):
	reqlist = []
	params = []

	params.append(period +'?apikey=' +api_key)
	params.append('assets=' +','.join(assets))
	params.append('start=' +times['start'])
	params.append('end=' +times['end'])
	params.append('csv=1')

	reqlist.append(api_url)
	reqlist.append('trmi')
	reqlist.append(version)
	reqlist.append('data')
	reqlist.append(group)
	reqlist.append('&'.join(params))

	return '/'.join(reqlist)

if __name__ == '__main__':
	main(sys.argv[1:])
