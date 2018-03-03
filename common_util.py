# Kevin Patel

import sys
from os import sep, path, makedirs
from os.path import dirname, basename, realpath, exists, isfile
import pandas as pd
from json import load
from functools import partial
from contextlib import contextmanager
from timeit import default_timer


# ********** SYSTEM SETTINGS **********
# Project Root and Subpackage paths
CRUNCH_DIR = dirname(dirname(realpath(sys.argv[0]))) +sep
RAW_DIR = CRUNCH_DIR +'raw' +sep
DATA_DIR = CRUNCH_DIR +'data' +sep
TRANSFORM_DIR = CRUNCH_DIR +'transform' +sep
EDA_DIR = CRUNCH_DIR +'eda' +sep

# Supported Pandas DF IO Formats
FMT_EXTS = {
	'csv': ('.csv',),
	'feather': ('.feather',),
	'hdf_fixed': ('.h5', '.hdf', '.he5', '.hdf5'),
	'hdf_table': ('.h5', '.hdf', '.he5', '.hdf5'),
	'parquet': ('.parquet',)
}

# Default Pandas DF IO format
DF_DATA_FMT = 'parquet'

# ********** CONSTANTS **********


# ********** FS AND GENERAL IO UTILS **********
get_script_dir = lambda: dirname(realpath(sys.argv[0])) +sep
get_parent_dir = lambda: dirname(dirname(realpath(sys.argv[0]))) +sep
makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not exists(dir_path) else None

def load_json(fname, dir_path=None):
	fpath = str(dir_path + fname) if dir_path else fname

	if (isfile(fpath)):
		with open(fpath) as json_data:
			return load(json_data)
	else:
		print(basename(fpath), 'must be in:', dirname(fpath))
		sys.exit(2)


# ********** PANDAS IO UTILS **********
def load_df(fname, dir_path=None, subset=None, data_format=DF_DATA_FMT):
	"""Assumes that source file has a non-default index column as the first column"""
	ext_tuple = FMT_EXTS[data_format]
	fpath = str(dir_path + fname) if dir_path else fname
	if (not fname.endswith(ext_tuple)):
		fpath += ext_tuple[0]

	if (isfile(fpath)):
		try:
			load_fn = {
				'csv': partial(pd.read_csv, index_col=0, usecols=subset),
				'feather': partial(pd.read_feather),
				'hdf_fixed': partial(pd.read_hdf, key=None, mode='r', columns=subset, format='fixed'),
				'hdf_table': partial(pd.read_hdf, key=None, mode='r', columns=subset, format='table'),
				'parquet': partial(pd.read_parquet, columns=subset)
			}.get(data_format)
			df = load_fn(fpath)
			return df.set_index('id') if data_format=='feather' else df

		except Exception as e:
			print('error during load:', e)
			sys.exit(2)
	else:
		print(basename(fpath), 'must be in:', dirname(fpath))
		sys.exit(2)

def dump_df(df, fname, dir_path=None, data_format=DF_DATA_FMT):
	ext_tuple = FMT_EXTS[data_format]
	fpath = str(dir_path + fname) if dir_path else fname
	if (not fname.endswith(ext_tuple)):
		fpath += ext_tuple[0]

	try:
		{
			'csv': df.to_csv,
			'feather': (lambda f: df.reset_index().to_feather(f)),
			'hdf_fixed': partial(df.to_hdf, fname, mode='w', format='fixed'),
			'hdf_table': partial(df.to_hdf, fname, mode='w', format='table'),
			'parquet': df.to_parquet
		}.get(data_format)(fpath)
	except Exception as e:
		print('error during dump:', e)
		sys.exit(2)


# ********** PANDAS UTILS **********
left_join = lambda a,b: a.join(b, how='left', sort=True)
right_join = lambda a,b: a.join(b, how='right', sort=True)
inner_join = lambda a,b: a.join(b, how='inner', sort=True)
outer_join = lambda a,b: a.join(b, how='outer', sort=True)


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


# ********** PROFILING UTILS **********
# The following class was written by stackoverflow's user bburns.km
# https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python/41408510#41408510
class benchmark(object):
	def __init__(self, msg, fmt="%0.3g"):
		self.msg = msg
		self.fmt = fmt

	def __enter__(self):
		self.start = default_timer()
		return self

	def __exit__(self, *args):
		t = default_timer() - self.start
		print(("%s : " + self.fmt + " seconds") % (self.msg, t))
		self.time = t
