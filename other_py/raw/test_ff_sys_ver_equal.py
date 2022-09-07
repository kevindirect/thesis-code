import pandas as pd
import json
import sys
from os import getcwd, sep, path, makedirs, pardir

def print_vers_equal(sent_df, group_name):
	# print(group_name +'_ver_N', sent_df[group_name +'_ver_N'].unique())
	# print(group_name +'_ver_S', sent_df[group_name +'_ver_S'].unique())
	# print(group_name +'_ver', sent_df[group_name +'_ver'].unique())

	# ver_N_ff = sent_df[group_name +'_ver_N'].fillna(axis=0, method='bfill')
	# ver_S_ff = sent_df[group_name +'_ver_S'].fillna(axis=0, method='bfill')
	# ver_ff = sent_df[group_name +'_ver'].fillna(axis=0, method='bfill')

	ver_N_ff = sent_df[group_name +'_ver_N']
	ver_S_ff = sent_df[group_name +'_ver_S']
	ver_ff = sent_df[group_name +'_ver']

	print('ver_N_ff.equals(ver_S_ff):', ver_N_ff.equals(ver_S_ff))
	print('ver_N_ff.equals(ver_ff):', ver_N_ff.equals(ver_ff))
	print('ver_S_ff.equals(ver_ff):', ver_S_ff.equals(ver_ff))

pfx = getcwd() +sep
pathsfile = 'paths.json'

# pathsfile tells script what to pull from the api and where to put it
if (path.isfile(pfx +pathsfile)):
	with open(pfx +pathsfile) as json_data:
		trmi_paths = json.load(json_data)['trmi']
else:
	print(pathsfile, 'must be present in the current directory')
	sys.exit(2)

for ver, groups in trmi_paths.items():
	print('ver:', ver)
	for group, assets in groups.items():
		print('group:', group)
		path = '.' +sep +'trmi' +sep +ver + sep +group +sep
		for asset in assets:
			print(asset)
			print_vers_equal(pd.read_csv(path +asset +'.csv'), group[-3:])
			print()
