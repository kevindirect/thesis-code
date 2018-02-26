# Kevin Patel

import sys
from os import sep, path, makedirs
from os.path import dirname, realpath, exists, isfile
import pandas as pd
from json import load


# ********** CONSTANTS **********
MONTH_NUM = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
			'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}

CRUNCH_DIR = dirname(dirname(realpath(sys.argv[0]))) +sep
RAW_DIR = CRUNCH_DIR +'raw' +sep
DATA_DIR = CRUNCH_DIR +'data' +sep
EDA_DIR = CRUNCH_DIR +'eda' +sep


# ********** OS AND IO UTILS **********
get_script_dir = lambda: dirname(realpath(sys.argv[0])) +sep
get_parent_dir = lambda: dirname(dirname(realpath(sys.argv[0]))) +sep
makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not exists(dir_path) else None

def load_json(fname, dir_path=get_script_dir()):
	full_path = dir_path +fname

	if (isfile(full_path)):
		with open(full_path) as json_data:
			return load(json_data)
	else:
		print(fname, 'must be present in the following directory:', dir_path)
		sys.exit(2)


# ********** PANDAS UTILS **********
left_join = lambda a,b: a.join(b, how='left', sort=True)
right_join = lambda a,b: a.join(b, how='right', sort=True)
inner_join = lambda a,b: a.join(b, how='inner', sort=True)
outer_join = lambda a,b: a.join(b, how='outer', sort=True)

def load_csv(fname, dir_path=get_script_dir(), idx_0=True, full_path_or_url=False):
	full_path = dir_path +fname

	if (full_path_or_url):
		# Also useful to load csv files form URLs
		return pd.read_csv(fname, index_col=0) if idx_0 else pd.read_csv(fname)
	else:
		if (isfile(full_path)):
			return pd.read_csv(full_path, index_col=0) if idx_0 else pd.read_csv(full_path)
		else:
			print(fname, 'must be present in the following directory:', dir_path)
			sys.exit(2)


# ********** MISC UTILS **********
# Selects a subset of str_list as dictated by qualifier_dict
# returns the subset that satisfies:
# IN(QUALIFIER_1 OR QUALIFIER_2 OR ... OR QUALIFIER_N-1) AND NOT IN(EXCLUDE_1 OR EXCLUDE_2 OR ... OR EXCLUDE_N-1)
def get_subset(str_list, qualifier_dict):
	selected = []

	selected.extend([string for string in str_list if string in qualifier_dict['exact']])
	selected.extend([string for string in str_list if string.startswith(tuple(qualifier_dict['startswith']))])
	selected.extend([string for string in str_list if string.endswith(tuple(qualifier_dict['endswith']))])
	selected.extend([string for string in str_list if any(re.match(rgx, string) for rgx in qualifier_dict['regex'])])
	if (qualifier_dict['exclude'] is not None):
		exclude_fn = lambda string: string not in get_subset(str_list, qualifier_dict['exclude'])
		selected = filter(exclude_fn, selected)

	return list(dict.fromkeys(selected)) # Remove dups (casting to dict keys retains order in Python 3.6+) and cast to list

