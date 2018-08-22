"""
System level settings/constants and common utilities for all crunch subpackages.
Kevin Patel
"""

import sys
from os import sep, path, makedirs, walk, listdir, rmdir
from os.path import dirname, basename, realpath, exists, isfile, getsize, join as path_join
from json import load, dump, dumps
import operator
import getopt
from difflib import SequenceMatcher
from collections import defaultdict, OrderedDict
from itertools import chain, tee
from functools import reduce, partial, wraps
from datetime import datetime
from timeit import default_timer
import logging

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, CustomBusinessHour
from pandas.testing import assert_series_equal, assert_frame_equal
from pandas.api.types import is_numeric_dtype


""" ********** SYSTEM SETTINGS ********** """
"""Project Root and Subpackage paths"""
logging.info('script location: ' +str(realpath(sys.argv[0])))
CRUNCH_DIR = dirname(dirname(realpath(sys.argv[0]))) +sep # FIXME
RAW_DIR = CRUNCH_DIR +'raw' +sep
DATA_DIR = CRUNCH_DIR +'data' +sep
MUTATE_DIR = CRUNCH_DIR +'mutate' +sep
RECON_DIR = CRUNCH_DIR +'recon' +sep
MODEL_DIR = CRUNCH_DIR +'model' +sep

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
EMPTY_STR = ''
JSON_SFX_LEN = len(".json")
DT_DAILY_FREQ = 'D'
DT_HOURLY_FREQ = 'H'
DT_CAL_DAILY_FREQ = DT_DAILY_FREQ
DT_BIZ_DAILY_FREQ = 'B'
DT_BIZ_HOURLY_FREQ = 'BH'
DT_FMT_YMD =  '%Y-%m-%d'
DT_FMT_YMD_HM = '%Y-%m-%d %H:%M'
DT_FMT_YMD_HMS = '%Y-%m-%d %H:%M:%S'

"""String"""
"""
Return string with escaped quotes enclosed around it.
Useful for programs, commands, and engines with text interfaces that use
enclosing quotes to recognize strings (like numexpr and sql).
"""
quote_it = lambda string: '\'' +string +'\''

wrap_parens = lambda string: '(' +string +')'

"""Datetime"""
dt_now = lambda: datetime.now()
str_now = lambda: dt_now().strftime(DT_FMT_YMD_HMS)

"""List"""
def remove_dups_list(lst):
	return list(OrderedDict.fromkeys(lst))

def flatten2D(list2D):
	return list(chain(*list2D))

def list_compare(master, other):
	"""
	Return describing relationship master and other.

	Args:
		master (list): 
		other (list): 
	
	Return:
		String describing relationship of lists
	"""
	master_set = set(master)
	other_set = set(other)

	if (master_set == other_set):
		return 'equal'
	elif (master_set > other_set):
		return 'proper_superset'
	elif (master_set < other_set):
		return 'proper_subset'
	elif (master_set & other_set == other_set):
		return 'has_all'
	elif (master_set & other_set < other_set):
		return 'has_some'
	elif (master_set.isdisjoint(other_set)):
		return 'disjoint'

def pairwise(iterable):
	"""
	Pairwise iterator (ie, size 2 sliding window).
	Taken from itertools recipes (official docs): https://docs.python.org/3/library/itertools.html

	"s -> (s0,s1), (s1,s2), (s2, s3), ..."
	"""
	a, b = tee(iterable)
	next(b, None)
	return zip(a, b)

def best_match(original_key, candidates, alt_maps=None):
	"""
	Return string from candidates that is the best match to the original key
	"""
	if (original_key in candidates):		# exact match
		return original_key
	elif(len(candidates)==1):				# unchanging
		return candidates[0]
	elif (alt_maps is not None):			# mapped match
		alt_keys = [original_key.replace(old, new) for old, new in alt_maps.items() if (old in original_key)]
		for alt_key in alt_keys:
			if (alt_key in candidates):
				return alt_key
	else:									# inexact longest subseq match
		match_len = [SequenceMatcher(None, original_key, can).find_longest_match(0, len(original_key), 0, len(can)).size for can in candidates]
		match_key = candidates[match_len.index(max(match_len))]
		logging.warn('using inexact match: ' +str(quote_it(original_key)) +' mapped to ' +str(quote_it(match_key)))
		return match_key

"""Dict"""
def nice_print_dict(dictionary):
	print(dumps(dictionary, indent=4, sort_keys=True))

def recursive_dict():
	"""
	Creates a recursive nestable defaultdict.

	In other words, it will automatically create intermediate keys if
	they don't exist!
	"""
	return defaultdict(recursive_dict)

def list_get_dict(dictionary, key_list):
    return reduce(operator.getitem, key_list, dictionary)

def list_set_dict(dictionary, key_list, value):
    list_get_dict(dictionary, key_list[:-1])[key_list[-1]] = value

def dict_path(dictionary, path=None, stop_cond=lambda v: not isinstance(v, dict)):
	"""
	Convenience function to give explicit paths from root keys until stop_cond is met.
	By default stop_cond is set such that the path to all leaves (non-dict values) are found.
	"""
	if (path is None):
		path = []
	for key, val in dictionary.items():
		newpath = path + [key]
		if (stop_cond(val)):
			yield newpath, val
		else:
			for unfinished in dict_path(val, newpath, stop_cond=stop_cond):
				yield unfinished

"""Math"""
def zdiv(top, bottom, zdiv_ret=0):
	return top/bottom if (bottom != 0) else zdiv_ret


""" ********** FS AND GENERAL IO UTILS ********** """
get_script_dir = lambda: dirname(realpath(sys.argv[0])) +sep
get_parent_dir = lambda: dirname(dirname(realpath(sys.argv[0]))) +sep
makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not exists(dir_path) else None

def load_json(fname, dir_path=None):
	fpath = str(dir_path + fname) if dir_path else fname

	if (isfile(fpath)):
		with open(fpath) as json_data:
			try:
				return load(json_data)
			except Exception as e:
				logging.error('error in file', str(fname +':'), str(e))
				raise e
	else:
		raise FileNotFoundError(str(basename(fpath) +' must be in:' +dirname(fpath)))

def dump_json(json_dict, fname, dir_path=None, ind="\t", seps=None, **kwargs):
	fpath = str(dir_path + fname) if dir_path else fname

	if (isfile(fpath)):
		logging.debug('json file exists at ' +str(fpath) +', syncing...')
	else:
		logging.debug('json file does not exist at ' +str(fpath) +', writing...')

	with open(fpath, 'w', **kwargs) as json_fp:
		try:
			return dump(json_dict, json_fp, indent=ind, separators=seps, **kwargs)
		except Exception as e:
			logging.error('error in file', str(fname +':'), str(e))
			raise e

def get_cmd_args(argv, arg_list, script_name=''):
	arg_list = ['help'] + arg_list 	# Add any commonly used args here
	arg_list_short = [str(arg_name[0] + ':' if arg_name[-1]=='=' else arg_name[0]) for arg_name in arg_list]
	arg_str = ''.join(arg_list_short)
	res = {arg_name: None for arg_name in arg_list}

	arg_list_short_no_sym = [arg_short[0] for arg_short in arg_list_short]
	assert(len({}.fromkeys(arg_list_short_no_sym)) == len(arg_list_short_no_sym))	# assert first letters of arg names are unique

	help_arg_strs = ['-{} <{}>'.format(arg_list_short_no_sym[i], arg_list[i][:-1] if arg_list[i][-1]=='=' else arg_list[i]) \
		for i in range(len(arg_list))]
	help_fn = lambda: print('{}.py {}'.format(script_name, str('[' +' '.join(help_arg_strs) +']')))

	try:
		opts, args = getopt.getopt(argv, str('h' +arg_str), list(['help']+arg_list))
	except getopt.GetoptError:
		help_fn()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			help_fn()
			sys.exit()
		else:
			for idx, arg_name in enumerate(arg_list):
				arg_char = arg_list_short[idx][0]

				if (arg_name[-1] == '='):
					if opt in (str('-'+arg_char), str('--'+arg_name[:-1])):
						res[arg_name] = arg
				else:
					if opt in (str('-'+arg_char), str('--'+arg_name)):
						res[arg_name] = True

	return res

def remove_empty_dirs(root_dir_path):
	for path, subdirs, files in walk(root_dir_path, topdown=False):
		for subdir in subdirs:
			dir_path = path_join(path, subdir)
			if not listdir(dir_path):  			# An empty list is False
				rmdir(path_join(path, subdir))


""" ********** PANDAS IO UTILS ********** """
def load_df(fname, dir_path=None, data_format=DF_DATA_FMT, subset=None, dti_freq=None):
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
			df = {
				'csv': partial(pd.read_csv, index_col=0, usecols=subset),
				'feather': partial(pd.read_feather),
				'hdf_fixed': partial(pd.read_hdf, key=None, mode='r', columns=subset, format='fixed'),
				'hdf_table': partial(pd.read_hdf, key=None, mode='r', columns=subset, format='table'),
				'parquet': partial(pd.read_parquet, columns=subset)
			}.get(data_format)(fpath)

			if (data_format=='feather'):
				df = df.set_index('id')

			if (dti_freq is not None):
				try:
					df = df.asfreq(dti_freq) 										# alt method (this adds null rows)
					# df.index.freq = pd.tseries.frequencies.to_offset(dti_freq) 	# Old way
				except Exception as ve:
					logging.warning('could not change time series index freq')
					logging.warning('ValueError caught:', ve)

			return df

		except Exception as e:
			logging.error('error during load:', e)
			raise e
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
		logging.error('error during dump:', e)
		raise e


""" ********** PANDAS GENERAL UTILS ********** """
left_join = lambda a,b: a.join(b, how='left', sort=True)
right_join = lambda a,b: a.join(b, how='right', sort=True)
inner_join = lambda a,b: a.join(b, how='inner', sort=True)
outer_join = lambda a,b: a.join(b, how='outer', sort=True)

"""Datetime"""
def series_to_dti(ser, fmt=DT_FMT_YMD_HM, utc=True, exact=True, freq=DT_HOURLY_FREQ):
	"""
	Return object (str) dtyped series as DatetimeIndex dtyped series.
	Sets the global project default for str -> DateTimeIndex conversion.
	"""
	dti = pd.to_datetime(ser, format=fmt, utc=utc, exact=exact)
	dti.freq = pd.tseries.frequencies.to_offset(freq)
	assert(np.all(dti.minute==0) and np.all(dti.second==0) and np.all(dti.microsecond==0) and np.all(dti.nanosecond==0))
	return dti

def get_missing_dt(ser, ref=DT_BIZ_DAILY_FREQ):
	"""
	Return the datetimes in ref that are missing from ser.
	"""
	biz_days = pd.date_range(ser.index.min(), ser.index.max(), freq=ref).date
	df_biz_days = ser.resample(ref).mean().dropna().index.date

	biz_days = pd.DatetimeIndex(biz_days)
	df_biz_days = pd.DatetimeIndex(df_biz_days)

	return biz_days.difference(df_biz_days)

def get_custom_biz_freq(ser, ref=DT_BIZ_DAILY_FREQ):
	"""
	XXX DEPRECATED
	Return custom CustomBusinessDay or CustomBusinessHour based on missing periods in ser.
	"""
	misses = get_missing_dt(ser, ref=ref)
	if (ref == DT_BIZ_DAILY_FREQ):
		return CustomBusinessDay(holidays=misses)
	elif (ref == DT_BIZ_HOURLY_FREQ):
		return CustomBusinessHour(holidays=misses)

def get_custom_biz_freq_ser(ser, ref=DT_BIZ_DAILY_FREQ):
	"""
	Return custom CustomBusinessDay or CustomBusinessHour based on missing periods in ser.
	"""
	misses = get_missing_dt(ser, ref=ref)
	if (ref == DT_BIZ_DAILY_FREQ):
		return CustomBusinessDay(holidays=misses)
	elif (ref == DT_BIZ_HOURLY_FREQ):
		return CustomBusinessHour(holidays=misses)

def get_custom_biz_freq_df(df, ref=DT_BIZ_DAILY_FREQ):
	"""
	Return custom CustomBusinessDay or CustomBusinessHour based on missing periods in each column of df
	as a dictionary of frequencies.
	"""
	cust_freqs = {}
	for column in df.columns:
		cust_freqs[column] = get_custom_biz_freq(df[column])
	return cust_freqs

def cust_count(df):
	"""
	Return custom biz freq and a Dataframe of counts per aggregation period.
	"""
	cust = get_custom_biz_freq(df)
	count_df = df.groupby(pd.Grouper(freq=DT_BIZ_DAILY_FREQ)).count()

	return cust, dti_to_ymd(count_df)

def get_time_mask(df, offset_col_name=None, offset_unit=DT_HOURLY_FREQ, offset_tz=None, time_range=None):
	"""
	Return a df of shifted, range-masked times.
	Setting start time temporally after end time filters for times outside of the time range.

	Args:
		df (pd.DataFrame): pd.DateTimeIndex indexed pd.DataFrame
		offset_col_name (str, Optional): name of offset column, if not given index is assumed to be in desired timezone
		offset_unit (str, Optional): unit of offset, must be supplied if offset_col_name is not None
		offset_tz (str, Optional): target timezone of offset
		time_range (list(str), Optional): start and end time range in form of strings with format options:
											'%H:%M', '%H%M', '%I:%M%p', '%I%M%p',
											'%H:%M:%S', '%H%M%S', '%I:%M:%S%p', '%I%M%S%p'

	Returns:
		pd.DataFrame indexed by original time with times column (shifted if an offset column was specified)
	"""
	mask_df = pd.DataFrame(data={'times': df.index}, index=df.index)

	if (offset_col_name is not None):
		lt_offset = pd.TimedeltaIndex(data=df.loc[:, offset_col_name], unit=offset_unit)
		mask_df['times'] = mask_df['times'] + lt_offset
		if (offset_tz is not None):
			mask_df['times'] = mask_df['times'].tz_convert(offset_tz)

	if (time_range is not None):
		indices = pd.DatetimeIndex(mask_df['times']).indexer_between_time(time_range[0], time_range[1])
		mask_df = mask_df.iloc[indices]

	return mask_df

"""Numpy"""
def pd_to_np(fn):
	"""
	Return function with all pandas typed arguments converted to their numpy counterparts.
	For use as a decorator.

	Args:
		fn (function): function to change the argument types of

	Returns:
		Functions with argument types casted
	"""
	def convert(obj):
		"""
		Return converted object
		
		Args:
			obj (Any): object to cast

		Returns:
			Converted object according to this map:
				pd.Series		-> np.array
				pd.DataFrame 	-> np.matrix
				Other -> other
		"""
		if (isinstance(obj, pd.Series)):
			return obj.values
		elif (isinstance(obj, pd.DataFrame)):
			return obj.as_matrix
		else:
			obj

	@wraps(fn)
	def fn_numpy_args(*args, **kwargs):
		args = tuple(map(convert, args))
		kwargs = {key: convert(val) for key, val in kwargs.items()}

		return fn(*args, **kwargs)

	return fn_numpy_args


""" ********** PANDAS DATAFRAME CHECKING UTILS ********** """
count_nn_df = lambda df: len(df) - df.isnull().sum()
count_nz_df = lambda df: df.apply(lambda ser: (ser.dropna(axis=0, how='any')!=0).sum())
count_nn_nz_df = lambda df: pd.concat([count_nn_df(df), count_nz_df(df)], axis=1, names=['non_nan', 'non_zero'])

def get_nmost_nulled_cols_df(df, n=5, keep_counts=False):
	nsmall = count_nn_df(df).nsmallest(n=n, keep='first')

	return nsmall if (keep_counts) else list(nsmall.index)

def is_empty_df(df, count_nans=False, how='all', **kwargs):
	if (count_nans):
		return df.empty
	else:
		return df.dropna(axis=0, how=how, **kwargs).empty

is_df = lambda candidate: isinstance(candidate, pd.DataFrame)
is_ser = lambda candidate: isinstance(candidate, pd.Series)

def assert_equal_pandas(df1, df2, **kwargs):
	are_frames = is_df(df1) and is_df(df2)
	are_series = is_ser(df1) and is_ser(df2)

	if (are_frames):
		assert_frame_equal(df1, df2, **kwargs)
	elif (are_series):
		assert_series_equal(df1, df2, **kwargs)


""" ********** PANDAS SEARCH AND FILTERING UTILS ********** """
""" DF Row Search """
"""Constants"""
DEFAULT_NUMEXPR = EMPTY_STR
DEFAULT_QUERY_JOIN = 'all'
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
		}.get(val[0], DEFAULT_NUMEXPR)

	elif (tup_len == 3):	# in range or out of range
		return {
			'in': str(val[1] +'<' +key +'<' +val[2]),
			'ine':str(val[1] +'<=' +key +'<=' +val[2]),
			'out': str(key +'<' +val[1] +' or ' +key +'>' +val[2]),
			'oute': str(key +'>=' +val[1] +' or ' +key +'<=' +val[2]),
		}.get(val[0], DEFAULT_NUMEXPR)

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
	}.get(type(val), DEFAULT_NUMEXPR)()

def build_query(search_dict, join_method=DEFAULT_QUERY_JOIN):
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

def chained_filter(str_list, qualifier_dict_list):
	"""
	Return subset in str_list that satisfies the list of qualifier_dicts via the procedure of get_subset.

	Args:
		str_list (list(str)): list of strings (ex: a list of column names)
		qualifier_dict_list (list(dict)): list of dictionaries representing a series of filters

	Returns:
		sublist of str_list
	"""
	if (not isinstance(qualifier_dict_list, list)):
		qualifier_dict_list = [qualifier_dict_list]

	return reduce(get_subset, qualifier_dict_list, str_list)


""" ********** DEBUGGING UTILS ********** """
in_debug_mode = lambda: logging.getLogger().isEnabledFor(logging.DEBUG)


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
			logging.info(("%s : " + self.fmt + " seconds") % (self.msg, t))
		self.time = t
