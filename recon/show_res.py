"""
Kevin Patel
"""
import sys
import os
import getopt
import logging

import numpy as np
import pandas as pd

from common_util import RECON_DIR, load_df
from recon.common import dum


def show_res(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	usage = lambda: print('show_res.py [-a <asset> -s <sfx>]')
	asset = 'sp_500'
	sfx = '_test'

	try:
		opts, args = getopt.getopt(argv, 'ha:s:', ['help', 'asset=', 'sfx='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-a', '--asset'):
			asset = arg
		elif opt in ('-s', '--sfx'):
			sfx = arg

	res_dfs = {
		'pba_oc_xwhole': load_df(str('pba_oc_return_fth_of_xwhole' +sfx +'.csv'), RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv'),
		'pba_oa_xwhole': load_df(str('pba_oa_return_fth_of_xwhole' +sfx +'.csv'), RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv'),
		'pba_oc_whole': load_df(str('pba_oc_return_fth_af_whole' +sfx +'.csv'), RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv'),
		'pba_oa_whole': load_df(str('pba_oa_return_fth_af_whole' +sfx +'.csv'), RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv')
	}

	print('asset:', asset)
	for res_df_name, res_df in res_dfs.items():
		print(res_df_name)
		print(res_df.groupby('label_name')[['best_score']].describe())

if __name__ == '__main__':
	show_res(sys.argv[1:])
