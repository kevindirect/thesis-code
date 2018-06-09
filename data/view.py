# Kevin Patel

import sys
import os
import getopt
from textwrap import indent
import logging

import numpy as np
import pandas as pd

from common_util import DATA_DIR, load_json, count_nn_df, count_nz_df, count_nn_nz_df, benchmark
from data.common import default_viewfile
from data.data_api import DataAPI

def view(argv):
	usage = lambda: print('view.py [-n -z -b -i -d -s -r -c -f <searchfile>]')
	viewfile = default_viewfile

	try:
		opts, args = getopt.getopt(argv, 'hnzbidsrc:f:', ['help', 'nonnan', 'nonzero', 'both', 'info', 'describe', 'show', 'random', 'count=', 'searchfile='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	
	num_rows = 20
	debugs_activated = []

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-n', '--nonnan'):      debugs_activated.append('n')
		elif opt in ('-z', '--nonzero'):     debugs_activated.append('z')
		elif opt in ('-b', '--both'):        debugs_activated.append('b')
		elif opt in ('-i', '--info'):        debugs_activated.append('i')
		elif opt in ('-d', '--describe'):    debugs_activated.append('d')
		elif opt in ('-s', '--show'):        debugs_activated.append('s')
		elif opt in ('-r', '--random'):      debugs_activated.append('r')
		elif opt in ('-c', '--count'):       num_rows = int(arg)
		elif opt in ('-f', '--searchfile'):  viewfile = arg

	view_search_dicts = load_json(viewfile, dir_path=DATA_DIR)

	debug_functions = {
		'n': count_nn_df,
		'z': count_nz_df,
		'b': count_nn_nz_df,
		'i': lambda df: df.info(verbose=True, null_counts=True),
		'd': lambda df: df.describe(),
		's': lambda df: df.head(num_rows),
		'r': lambda df: df.sample(n=num_rows, axis=0),
	}

	for key, search_dict in view_search_dicts.items():
		print('group:', key)

		for rec, gen_df in DataAPI.generate(search_dict):
			print('root:', rec.root)
			print('desc:', rec.desc)
			print('\n')

			for activated in debugs_activated:
				print('debug:', activated)
				print(debug_functions[activated](gen_df))
				print('\n')

		print('\n')

	
if __name__ == '__main__':
	view(sys.argv[1:])