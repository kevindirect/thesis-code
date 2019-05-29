"""
Kevin Patel
"""
import sys
import getopt
from os import sep

from common_util import RAW_DIR, load_json, makedir_if_not_exists, dump_df
from raw.common import TRMI_CONFIG_FNAME, TRMI_CONFIG_DIR, default_pathsfile, load_csv_no_idx


def get_trmi(argv):
	usage = lambda: print('get_trmi.py [-p <pathsfile> -k -t]')

	# trmi.json file contains api url and key
	trmi = load_json(TRMI_CONFIG_FNAME, dir_path=TRMI_CONFIG_DIR)

	# Default Parameters
	per = 'hourly'   			# daily, hourly, or minutely
	pathsfile = default_pathsfile
	keep_ns = False
	is_test = False
	startend = {'start': '1996-01-01', 'end':'2018-01-10'}

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
			is_test = True
			startend = {'start': '2015-08-01', 'end':'2015-08-03'}

	# pathsfile tells script what to pull from the api and where to put it
	trmi_paths = load_json(pathsfile, dir_path=RAW_DIR)['trmi']

	dropfirst = ['id', 'Date', 'Asset']
	join_cols = ['assetCode', 'windowTimestamp']
	for ver, groups in trmi_paths.items():
		for group, assets in groups.items():
			endpoint = make_csv_group_request_url(group, assets, ver, per, startend, trmi['api']['url'], trmi['api']['key'])
			print(group, endpoint)
			df = load_csv_no_idx(endpoint, local_csv=False)
			target_dir = RAW_DIR +'trmi' +sep +ver +sep +group +sep
			makedir_if_not_exists(target_dir)

			df = df.drop(dropfirst, axis=1, errors='ignore')
			news = df.loc[df['dataType'] == 'News'].drop('dataType', axis=1)
			social = df.loc[df['dataType'] == 'Social'].drop('dataType', axis=1)
			merged = news.merge(social, on=join_cols, suffixes=('_N', '_S'))
			if (keep_ns):
				news_social = df.loc[df['dataType'] == 'News_Social'].drop('dataType', axis=1)
				merged = merged.merge(news_social, on=join_cols, suffixes=('', '_NS'))

			# Shorten systemVersion column names
			merged.rename(columns={'systemVersion_N': 'ver_N', 'systemVersion_S': 'ver_S', 'systemVersion': 'ver'}, inplace=True)

			# Simple cleaning of system version to make formatting consistent
			merged['ver_N'] = merged['ver_N'].str.replace('MP:', '', case=False)
			merged['ver_S'] = merged['ver_S'].str.replace('MP:', '', case=False)
			if (keep_ns):
				merged['ver'] = merged['ver'].str.replace('MP:', '', case=False)

			# Remove redundant system version columns
			if (merged['ver_N'].equals(merged['ver_S'])):
				merged = merged.drop('ver_S', axis=1, errors='ignore')
				if (keep_ns):
					if (merged['ver_N'].equals(merged['ver'])):
						merged = merged.drop('ver', axis=1, errors='ignore')
						merged.rename(columns={'ver_N': 'ver'}, inplace=True)
				else:
					merged.rename(columns={'ver_N': 'ver'}, inplace=True)

			# Last step before splitting: prefix all data columns with last three letters of group name and trmi version
			merged.columns = merged.columns.map(lambda s: str(group[-3:] +'_' +ver +'_' +s) if s not in join_cols else s)

			for asset in assets:
				print('\t' +asset, end='...', flush=True)
				asset_df = merged.loc[merged['assetCode'] == asset]
				asset_df.insert(0, 'id', asset_df['windowTimestamp'].map(lambda w: w[:16]))
				asset_df = asset_df.drop(join_cols, axis=1)
				asset_df = asset_df.set_index('id', drop=True).sort_index()
				asset += '_test' if (is_test) else ''
				dump_df(asset_df, asset, dir_path=target_dir)
				print('done')
			print()


def make_csv_group_request_url(group, assets, version, period, times, api_url, api_key):
	"""TRMI API ver 1.1"""
	reqlist, params = [], []

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
	get_trmi(sys.argv[1:])
