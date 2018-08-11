# Kevin Patel

import sys
import os
import os.path
import logging

import numpy as np
import pandas as pd

from common_util import DATA_DIR, DF_DATA_FMT, in_debug_mode, remove_empty_dirs, get_cmd_args, load_json
from data.common import default_cleanfile
from data.data_api import DataAPI

def clean(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
	cmd_arg_list = ['cleanfile=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name='runt')
	cleanfile = cmd_input['cleanfile='] if (cmd_input['cleanfile='] is not None) else default_cleanfile

	clean = load_json(cleanfile, dir_path=DATA_DIR)
	whitelist_df = DataAPI.get_record_view().loc[:, ['name', 'dir']]
	file_ext = '.' +clean['ext'] if (clean['ext'] != 'default') else '.' +DF_DATA_FMT

	logging.info('building blacklists...')
	blacklists = {}
	for clean_dir in clean['clean_dirs']:
		blacklists[clean_dir] = []
		for path, subdirs, fnames in os.walk(DATA_DIR +clean_dir):
			for fname in fnames:
				fname_no_ext, fname_ext = os.path.splitext(fname)
				if ((clean['check_ext'] and fname_ext==file_ext) and not any(whitelist_df['name'].str.match(fname_no_ext))):
					blacklists[clean_dir].append(path +os.sep +fname)

		if (in_debug_mode()):
			logging.debug(clean_dir +' blacklist: ' +'\n' +str('\n'.join(blacklists[clean_dir])))

	logging.info('removing files in blacklists...')			
	for root_name in blacklists:
		logging.debug(root_name)
		for path in blacklists[root_name]:
			os.remove(path)

	logging.info('cleaning up empty subdirectories...')
	for clean_dir in clean['clean_dirs']:
		remove_empty_dirs(DATA_DIR +clean_dir)


if __name__ == '__main__':
	clean(sys.argv[1:])