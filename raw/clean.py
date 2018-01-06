# Kevin Patel

import pandas as pd
import numpy as np
import datetime
import sys
import getopt
from os import getcwd, sep, path, makedirs
import json
sys.path.insert(0, path.abspath('..'))
from common import *

def main(argv):
	usage = lambda: print('clean.py -i <infile> [-c <cleanfile>]')
	pfx = getcwd() +sep
	dirty_dir = 'dirty' +sep
	clean_dir = 'clean' +sep
	clean_instr = 'clean.json'
	in_fname = None
	in_path = None
	frame = None
	out_fname = None

	try:
		opts, args = getopt.getopt(argv,'hi:c:',['help', 'infile=', 'cleanfile='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt == ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-i', '--infile'):
			in_fname = arg
			in_path = pfx +dirty_dir +in_fname
			if (path.isfile(in_path)):
				frame = pd.read_csv(in_path, index_col=0)
				out_fname = arg
			else:
				sys.exit(2)
		elif opt in ('-c', '--cleanfile'):
			clean_instr = arg # Use an alternate json cleanfile (optional)

	if (frame is None):
		print('need to specify a valid dataframe to clean')
		usage()
		sys.exit(2)

	makedir_if_not_exists(pfx +clean_dir)
	print('cleaning ' +in_fname, end='...')
	with open(pfx +clean_instr) as json_data:
		clean_dict = json.load(json_data)
		if (in_fname in clean_dict):
			tasks = clean_dict[in_fname]
			if ("drop" in tasks):
				frame.drop(tasks["drop"], axis=1, inplace=True, errors='ignore')
			if ("rename" in tasks):
				frame.rename(columns=tasks["rename"], inplace=True)
			frame.to_csv(pfx +clean_dir +out_fname)
			print('done')
		else:
			print('no instructions for ' +in_fname)

if __name__ == '__main__':
	main(sys.argv[1:])
