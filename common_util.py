"""
System level settings/constants and common utilities for all crunch subpackages.
Kevin Patel
"""

import sys
from os import sep, path, makedirs
from os.path import dirname, basename, realpath, exists, isfile, getsize
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_list_like
from json import load
from functools import partial
from datetime import datetime
from contextlib import contextmanager
from timeit import default_timer


""" ********** SYSTEM SETTINGS ********** """
"""Project Root and Subpackage paths"""
CRUNCH_DIR = dirname(dirname(realpath(sys.argv[0]))) +sep
RAW_DIR = CRUNCH_DIR +'raw' +sep
DATA_DIR = CRUNCH_DIR +'data' +sep
TRANSFORM_DIR = CRUNCH_DIR +'transform' +sep
EDA_DIR = CRUNCH_DIR +'eda' +sep

"""Supported Pandas DF IO Formats"""
FMT_EXTS = {
	'csv': ('.csv',),
	'feather': ('.feather',),
	'hdf_fixed': ('.h5', '.hdf', '.he5', '.hdf5'),
	'hdf_table': ('.h5', '.hdf', '.he5', '.hdf5'),
	'parquet': ('.parquet',)
}

"""Default Pandas DF IO format"""
DF_DATA_FMT = 'parquet'


""" ********** GENERAL UTILS ********** """
"""Constants"""
BYTES_PER_MEGABYTE = 10**6
EMPTY_STR=''

"""String"""
"""
Return string with escaped quotes enclosed around it.
Useful for programs, commands, and engines with text interfaces that use
enclosing quotes to recognize strings (like numexpr and sql).
"""
quote_it = lambda string: '\'' +string +'\''

"""Datetime"""
dt_now = lambda: datetime.now()
str_now = lambda: dt_now().strftime('%Y-%m-%d %H:%M:%S')


""" ********** FS AND GENERAL IO UTILS ********** """
get_script_dir = lambda: dirname(realpath(sys.argv[0])) +sep
get_parent_dir = lambda: dirname(dirname(realpath(sys.argv[0]))) +sep
makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not exists(dir_path) else None

def load_json(fname, dir_path=None):
	fpath = str(dir_path + fname) if dir_path else fname

	if (isfile(fpath)):
		with open(fpath) as json_data:
			return load(json_data)
	else:
		raise FileNotFoundError(str(basename(fpath) +' must be in:' +dirname(fpath)))


""" ********** PANDAS IO UTILS ********** """
def load_df(fname, dir_path=None, subset=None, data_format=DF_DATA_FMT):
	"""
	Read and return the df file in the given directory and
	assume that the file has an index as the first column
	"""
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
		raise FileNotFoundError(str(basename(fpath) +' must be in:' +dirname(fpath)))

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
		return getsize(fpath) // BYTES_PER_MEGABYTE

	except Exception as e:
		print('error during dump:', e)
		sys.exit(2)


""" ********** PANDAS GENERAL UTILS ********** """
left_join = lambda a,b: a.join(b, how='left', sort=True)
right_join = lambda a,b: a.join(b, how='right', sort=True)
inner_join = lambda a,b: a.join(b, how='inner', sort=True)
outer_join = lambda a,b: a.join(b, how='outer', sort=True)


""" ********** PANDAS SEARCH AND FILTERING UTILS ********** """
""" DF Row Search """
"""Constants"""
DEF_NUMEXPR = EMPTY_STR
DEF_QUERY_JOIN = 'all'
ALLOWED_TYPES = [int, float, str, tuple, list]

def equals_numexpr(key, val):
	"""Return numexpr expression for 'key equals value'"""
	return str(key +'==' +val)

# All of these have the same numexpr syntax hah
int_numexpr = equals_numexpr
float_numexpr = equals_numexpr
str_numexpr = equals_numexpr
list_numexpr = equals_numexpr

def tuple_numexpr(key, val):
	"""
	Use a string key and tuple value to create a numexpr
	string representing an inequality expression and return it.
	"""
	tup_len = len(val)
	val = tuple(map(str, val))

	if (tup_len == 2): 	# less than or greater than
		return {
			'lt': str(key +'<' + val[1]),
			'lte': str(key +'<=' + val[1]),
			'gt': str(key +'>' + val[1]),
			'gte': str(key +'>=' + val[1])
		}.get(val[0], DEF_NUMEXPR)

	elif (tup_len == 3):	# in range or out of range
		return {
			'in': str(val[1] +'<' +key +'<' +val[2]),
			'ine':str(val[1] +'<=' +key +'<=' +val[2]),
			'out': str(key +'<' +val[1] +' or ' +key +'>' +val[2]),
			'oute': str(key +'>=' +val[1] +' or ' +key +'<=' +val[2]),
		}.get(val[0], DEF_NUMEXPR)

def to_numexpr(key, val):
	"""
	Parse a key, val pair into a numexpr string and return it.
	Dispatcher function for build_query.
	# XXX - add type check assertions of val at some point
	"""
	return {
		int: partial(int_numexpr, key, str(val)),
		float: partial(float_numexpr, key, str(val)),
		str: partial(str_numexpr, key, quote_it(str(val))),
		list: partial(list_numexpr, key, str(val)),
		tuple: partial(tuple_numexpr, key, val)
	}.get(type(val), DEF_NUMEXPR)()

def build_query(search_dict, join_method=DEF_QUERY_JOIN):
	"""
	Return a numexpr query defined by search_dict.
	The join_method param sets how terms will be interpreted as a
	set of row filters (AND, OR, None (returns unjoined list)).
	The search_dict parameter has the following required format:
		{
			'$COL_NAME_1': TERM_1,
			'$COL_NAME_2': TERM_2,
			...
		}
	The supported filtering term types are:
		- numeric: exact match
		- string: exact match
		- list(string...): subset match
		- list(numeric...): subset match
		- tuple(string, numeric) or tuple(string, numeric, numeric): range match
	Currently no support for pandas series made of listlike objects. Currently
		df.query has no way of facillitating the search of a listlike series.
	The reason the format is defined thusly instead of using a
	(more complex) nested dict structure is because:
		1. Each df column (series) can only have one dtype anyway
		2. Flat is better than nested
		3. This format is good enough for most cases
		4. More complicated queries composed using multiple build_query calls (use join_method=None)
	"""
	subqueries = [to_numexpr(col_name, search_term) for col_name, search_term in search_dict.items()]
	return {
		'all': ' and '.join(subqueries),
		'any': ' or '.join(subqueries),
		None: subqueries
	}.get(join_method)

def query_df(df, numexpr):
	"""Return index of rows which match numexpr query"""
	return df.query(numexpr).index

def search_df(df, search_dict):
	"""Return index of rows which match search_dict"""
	assert((key in df.columns) for key in search_dict.keys())
	return query_df(df, build_query(search_dict))

""" DF Column Filter  """
def get_subset(str_list, qualifier_dict):
	"""
	Select a subset of str_list as dictated by qualifier_dict and return the subset, as list, that satisfies:
	IN(QUALIFIER_1 OR QUALIFIER_2 OR ... OR QUALIFIER_N-1) AND NOT IN(EXCLUDE_1 OR EXCLUDE_2 OR ... OR EXCLUDE_N-1)
	"""
	selected = []

	selected.extend([string for string in str_list if string in qualifier_dict['exact']])
	selected.extend([string for string in str_list if string.startswith(tuple(qualifier_dict['startswith']))])
	selected.extend([string for string in str_list if string.endswith(tuple(qualifier_dict['endswith']))])
	selected.extend([string for string in str_list if any(re.match(rgx, string) for rgx in qualifier_dict['regex'])])
	if (qualifier_dict['exclude'] is not None):
		exclude_fn = lambda string: string not in get_subset(str_list, qualifier_dict['exclude'])
		selected = filter(exclude_fn, selected)

	return list(dict.fromkeys(selected)) # Remove dups (casting to dict keys retains order in Python 3.6+) and cast to list


""" ********** PROFILING UTILS ********** """
# The following class was written by stackoverflow's user bburns.km
# https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python/41408510#41408510
class benchmark(object):
	def __init__(self, msg, fmt="%0.3g", suppress=False):
		self.msg = msg
		self.fmt = fmt
		self.suppress = suppress

	def __enter__(self):
		self.start = default_timer()
		return self

	def __exit__(self, *args):
		t = default_timer() - self.start
		if (not self.suppress):
			print(("%s : " + self.fmt + " seconds") % (self.msg, t))
		self.time = t
