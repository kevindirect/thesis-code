"""
System level settings/constants and common utilities for all crunch subpackages.
Kevin Patel
"""

import sys
from os import sep, path, makedirs, walk, listdir, rmdir
from os.path import dirname, basename, realpath, normpath, exists, isfile, getsize, join as path_join
import socket
from json import load, dump, dumps
import math
import numbers
import operator
import getopt
import collections.abc
from contextlib import suppress
from difflib import SequenceMatcher
from collections import defaultdict, MutableMapping, OrderedDict, ChainMap
from itertools import product, chain, tee, islice, chain, zip_longest
from functools import reduce, partial, wraps
from datetime import datetime, date, timedelta
from timeit import default_timer
import logging

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, CustomBusinessHour
from pandas.testing import assert_series_equal, assert_frame_equal
from pandas.api.types import is_numeric_dtype
import humanize


""" ********** SYSTEM SETTINGS ********** """
"""Project Root and Subpackage paths"""
CRUNCH_DIR = dirname(dirname(realpath(sys.argv[0]))) +sep # FIXME
RAW_DIR = CRUNCH_DIR +'raw' +sep
DATA_DIR = CRUNCH_DIR +'data' +sep
MUTATE_DIR = CRUNCH_DIR +'mutate' +sep
RECON_DIR = CRUNCH_DIR +'recon' +sep
MODEL_DIR = CRUNCH_DIR +'model' +sep
REPORT_DIR = CRUNCH_DIR +'report' +sep

logging.warning('script location: {}'.format(str(realpath(sys.argv[0]))))
logging.warning('using project dir: {}'.format(CRUNCH_DIR))

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


""" ********** SYSTEM UTILS ********** """
get_pardir_from_path = lambda path: basename(normpath(path))
add_sep_if_none = lambda path: path if (path[-1] == sep) else path+sep


""" ********** GENERAL UTILS ********** """
"""Constants"""
BYTES_PER_MEGABYTE = 10**6
EMPTY_STR = ''
JSON_SFX = '.json'
JSON_SFX_LEN = len(JSON_SFX)
DT_DAILY_FREQ = 'D'
DT_HOURLY_FREQ = 'H'
DT_CAL_DAILY_FREQ = DT_DAILY_FREQ
DT_BIZ_DAILY_FREQ = 'B'
DT_BIZ_HOURLY_FREQ = 'BH'
DT_FMT_YMD =  '%Y-%m-%d'
DT_FMT_YMD_HM = '%Y-%m-%d %H:%M'
DT_FMT_YMD_HMS = '%Y-%m-%d %H:%M:%S'
DT_FMT_YMD_HMSF = '%Y-%m-%d %H:%M:%S:%f'

"""Type"""
def is_type(obj, *types):
	return any([isinstance(obj, tp) for tp in types])

def isnt(obj):
	return is_type(obj, NoneType)

def is_real_num(obj):
	return is_type(obj, numbers.Real)

def is_seq(obj):
	return is_type(obj, collections.abc.Sequence)

def is_df(obj):
	return is_type(obj, pd.DataFrame)

def is_ser(obj):
	return is_type(obj, pd.Series)

def get_class_name(obj):
	"""
	Returns the class name of an object.
	"""
	return obj.__class__.__name__

"""String"""
"""
Return string with escaped quotes enclosed around it.
Useful for programs, commands, and engines with text interfaces that use
enclosing quotes to recognize strings (like numexpr and sql).
"""
quote_it = lambda string: '\'' +string +'\''

wrap_parens = lambda string: '(' +string +')'

def str_to_list(string, delimiter=',', cast_to=str):
	return list(map(cast_to, map(str.strip, string.split(delimiter))))

"""Datetime"""
dt_now = lambda: datetime.now()
str_now = lambda fmt=DT_FMT_YMD_HMS: dt_now().strftime(fmt)
dt_delta = lambda start, end: datetime.combine(date.min, end) - datetime.combine(date.min, start)

"""List"""
def remove_dups_list(lst):
	return list(OrderedDict.fromkeys(lst))

def flatten2D(list2D):
	return list(chain(*list2D))

def all_equal(lst):
	first_item = lst[0]
	return all(element==first_item for element in lst)

first_element = lambda lst: lst[0]

def get0(lst):
	"""
	Return first element if the list has length one, else return the list.
	"""
	if (len(lst) == 1):
		return lst[0]
	else:
		return lst

def getcon(lst, string):
	"""
	Return sublist of items containing string, if only one match return it as a singleton.
	"""
	return get0(list(filter(lambda el: string in el, lst)))

def list_compare(master, other):
	"""
	Return describing relationship master and other.

	Args:
		master (list): 
		other (list): 
	
	Returns:
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

def get_range_cuts(start, end, ratios_list):
	"""
	Return a list of segment indices for cuts based on ratios over the range provided by the passed [start, end).
	The cuts will traverse the whole range of [start, end), if the provided ratios sum to less than one
	the last segment will contain the remainder.
	"""
	cuts = [start]
	size = end - start
	seg_start = start

	for seg_ratio in ratios_list[:-1]:
		seg_end = seg_start + int(math.floor(seg_ratio*size))
		cuts.append(seg_end)
		seg_start = seg_end
	cuts.append(end)

	return cuts

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
	elif(len(candidates) == 1):				# unchanging
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
class NestedDefaultDict(MutableMapping):
	"""
	Nested Default Dictionary class.
	Defines a nested dictionary where arbitrary key lists are accomodated by instantiating default_dicts if that key list does not exist.
	Implements a dict-like interface.

	Will not hold a NestedDefaultDict as a value, if this is attempted the other NestedDefaultDict will be grafted to this one.
	Empty NestedDefaultDict objects cannot be grafted on to this one.
	"""
	KEY_END = '.'

	def __init__(self, keychains=None, tree=None, *args, **kwargs):
		"""
		NDD constructor.

		Args:
			keychains (list): paths to all leaves in tree
			tree (defaultdict): value tree, recursive defaultdict of defaultdicts
		"""
		recursive_dict = lambda: defaultdict(recursive_dict)
		self.keychains = [] if (keychains is None) else keychains
		self.tree = recursive_dict() if (keychains is None) else tree

	def empty(self):
		"""
		Return whether or not NDD is empty.
		"""
		return len(self.keychains) == 0

	def keys(self):
		"""
		Yield from sorted iterator of keychains.
		"""
		yield from sorted(self.keychains)

	def values(self):
		"""
		Yield from values in order of keychains.
		"""
		for key in self.keys():
			yield self.__getitem__(key)

	def items(self):
		"""
		Yield from key value pairs in order of keychains.
		"""
		for key in self.keys():
			yield key, self.__getitem__(key)

	def childkeys(self, key):
		"""
		Yield all child keychains of a keychain.
		This will also return the original key if it exists in the set of keychains (non-proper superset).
		"""
		yield from filter(lambda k: k[:len(key)]==key, self.keys())

	def __setitem__(self, key, value):
		"""
		Set an item in the object.
		If the value to set is a NestedDefaultDict, then it will be grafted on at the specified location,
		overwriting the old branch.

		Args:
			key (list): list of keys
			value (any): value to set

		Returns:
			None

		Raises:
			ValueError if the proposed key contains a reserved string
		"""
		if (NestedDefaultDict.KEY_END in key):
			raise ValueError("Cannot use \'{}\' in a valid keychain, this string is reserved".format(NestedDefaultDict.KEY_END))

		if (isinstance(value, NestedDefaultDict) or isinstance(value, defaultdict)):
			for childkey in self.childkeys(key):										# Remove old branch
				self.__delitem__(childkey)
			reduce(operator.getitem, key[:-1], self.tree)[key[-1]] = value.tree 		# Graft other NDD
			for k, v in value.items():
				self.__setitem__(key+k, v)
		else:
			reduce(operator.getitem, key, self.tree)[NestedDefaultDict.KEY_END] = value
			if (not key in self.keychains):
				self.keychains.append(key)

	def __getitem__(self, key):
		"""
		Get an item.

		Args:
			key (list): list of keys

		Returns:
			item

		Raises:
			ValueError if the key doesn't exist
		"""
		if key in self.keychains:
			return reduce(operator.getitem, key, self.tree)[NestedDefaultDict.KEY_END]
		else:
			raise ValueError("Attempted key doesn\'t exist")

	def __delitem__(self, key):
		"""
		Delete an item.
		Only deletes that exact key and item: if ['a', 'b', 'c'] and ['a', 'b', 'c', 'd'] exists and the key ['a', 'b', 'c'] is deleted,
		then ['a', 'b', 'c'] will continue to exist.

		Args:
			key (list): list of keys

		Returns:
			None

		Raises:
			ValueError if the key doesn't exist
		"""
		if key in self.keychains:
			del reduce(operator.getitem, key, self.tree)[NestedDefaultDict.KEY_END]
			self.keychains.remove(key)
		else:
			raise ValueError("Attempted key doesn\'t exist")

	def __iter__(self):
		"""
		Return iterator.
		"""
		return self.items()

	def __len__(self):
		"""
		Return number of valid keychains.
		"""
		return len(self.keychains)

	def __str__(self):
		"""
		Returns string representation
		"""
		return str(dumps(self.tree, indent=4, sort_keys=True))

	def __repr__(self):
		"""
		Echoes class, id, & reproducible representation in the REPL
		XXX - probably wrong
		"""
		return "{}, {}".format(self.keychains, self.tree)

def dict_combine(a, b):
	"""
	Combine / merge two dicts into one.
	"""
	return {**a, **b}

def nice_print_dict(dictionary):
	print(dumps(dictionary, indent=4, sort_keys=True))

def remove_keys(dictionary, list_keys):
	for key in list_keys:
		with suppress(KeyError):
			del dictionary[key]

	return dictionary

def recursive_dict():
	"""
	XXX - Deprecated in favor of NestedDefaultDict
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

def get_variants(mappings, fmt='grid'):
	"""
	Return all possible combinations of key-value maps as a list of dictionaries.

	{
		a: [1, 2, 3],
		b: [4, 5, 6],
	}

	would be mapped to

	[
		{a: 1, b: 4}, {a: 2, b: 4}, {a: 3, b: 4},
		{a: 1, b: 5}, {a: 2, b: 5}, {a: 3, b: 5},
		{a: 1, b: 6}, {a: 2, b: 6}, {a: 3, b: 6}
	]
	"""
	if (fmt == 'grid'):
		names, combos = list(mappings.keys()), list(product(*mappings.values()))
		variants = [{names[idx]: value for idx, value in enumerate(combo)} for combo in combos]
	elif (fmt == 'list'):
		pass # XXX - Implement

	return variants

"""Function"""
def compose(*fns):
	"""
	Perform function composition of passed functions, performed on input in the order they are passed.
	"""
	def composed(*args, **kwargs):
		val = fns[0](*args, **kwargs)
		for fn in fns[1:]:
			val = fn(val)
		return val

	return composed

def dcompose(*fns):
	"""
	Perform delayed function composition of passed functions, performed on input in the order they are passed.
	"""
	def dcomposed(*args, **kwargs):
		val = delayed(fns[0])(*args, **kwargs)
		for fn in fns[1:]:
			val = delayed(fn)(val)
		return val

	return dcomposed

"""Iterator"""
def group_iter(iterable, n=2, fill_value=None):
	"""
	Iterates over fixed length, non-overlapping windows
	"""
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	yield from zip_longest(*args, fillvalue=fill_value)

def window_iter(iterable, n=2):
	"""Returns a sliding window (of width n) over data from the iterable"""
	"""	s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ... """
	it = iter(iterable)
	result = tuple(islice(it, n))
	if len(result) == n:
		yield result
	for elem in it:
		result = result[1:] + (elem,)
		yield result

def col_iter(two_d_list):
	"""
	Iterates over columns of a two dimensional list
	"""
	yield from group_iter(chain.from_iterable(zip(*two_d_list)), n=len(two_d_list))

"""String Mappers"""
"""
The following are functions that return string mapping functions based on rule and handling parameters.
String mapping functions map a sequence of strings to a single string. Useful for naming new data columns.
"""
def concat_map(delimiter='_', **kwargs):
	return lambda *strings: delimiter.join(strings)

first_letter_concat = lambda lst: "".join((string[0] for string in lst))

def substr_ad_map(check_fn=all_equal, accord_fn=first_element, discord_fn=first_letter_concat, delimiter='_', **kwargs):
	"""
	Map a sequence of strings to one string by handling accordances or discordances in substrings.
	Assumes all strings in the sequence have an equal number of delimited substrings.
	"""
	def mapper(*strings):
		output = []
		str_row_vectors = [string.split(delimiter) for string in strings]

		for col in col_iter(str_row_vectors):
			substr = accord_fn(col) if (check_fn(col)) else discord_fn(col)
			output.append(substr)

		return delimiter.join(output)

	return mapper

"""Math"""
def zdiv(top, bottom, zdiv_ret=0):
	return top/bottom if (bottom != 0) else zdiv_ret

def apply_nz_nn(fn):
	"""
	Return modified function where fn is only applied if the value is non zero and non null.
	"""
	def func(val):
		if (val is None or val == 0):
			return val
		else:
			return fn(val)
	return func

one_minus = lambda val: 1 - val
identity_fn = lambda val, *args, **kwargs: val
null_fn = lambda *args, **kwargs: None


""" ********** FS AND GENERAL IO UTILS ********** """
get_script_dir = lambda: dirname(realpath(sys.argv[0])) +sep
get_parent_dir = lambda: dirname(dirname(realpath(sys.argv[0]))) +sep
makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not exists(dir_path) else None

def load_json(fname, dir_path=None):
	fpath = str(add_sep_if_none(dir_path) + fname) if dir_path else fname
	if (not fname.endswith(JSON_SFX)):
		fpath += JSON_SFX

	if (isfile(fpath)):
		with open(fpath) as json_data:
			try:
				return load(json_data)
			except Exception as e:
				logging.error('error in file', str(fname +':'), str(e))
				raise e
	else:
		raise FileNotFoundError(str(basename(fpath) +' must be in: ' +dirname(fpath)))

def dump_json(json_dict, fname, dir_path=None, ind="\t", seps=None, **kwargs):
	fpath = str(add_sep_if_none(dir_path) + fname) if dir_path else fname
	if (not fname.endswith(JSON_SFX)):
		fpath += JSON_SFX

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

def get_cmd_args(argv, arg_list, script_name='', set_logging=True):
	"""
	Parse commandline arguments from argv and return them as a dict.

	Args:
		argv (sys.argv): system argument input vector
		arg_list (list): list of non-static commandline arguments (end with '=' for non-flag arguments)
		script_name (str): name of calling script for use in the help dialog
		set_logging (bool): whether or not to include a logging level commandline argument and initialize logging

	Returns:
		Dict of commandline argument to value mappings, a value maps to None if arg was not set or flag argument was not raised
	"""
	static_args = ['help', 'loglevel='] if (set_logging) else ['help']
	arg_list = static_args + arg_list
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

	if (set_logging):
		set_loglevel(res['loglevel='])

	return res

def remove_empty_dirs(root_dir_path):
	for path, subdirs, files in walk(root_dir_path, topdown=False):
		for subdir in subdirs:
			dir_path = path_join(path, subdir)
			if not listdir(dir_path):  			# An empty list is False
				rmdir(path_join(path, subdir))

def get_free_port(host="localhost"):
	"""
	Get a free port on the machine.
	From the MongoBox project: https://github.com/theorm/mongobox
	"""
	temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	temp_sock.bind((host, 0))
	port = temp_sock.getsockname()[1]
	temp_sock.close()
	del temp_sock
	return port


""" ********** NUMPY GENERAL UTILS ********** """
def filter_null(arr):
	"""
	Return numpy array with all nulls removed.
	Also with pandas series.
	"""
	return arr[~pd.isnull(arr)]		# Filter None, NaN, and NaT values

def arr_nonzero(arr, ret_idx=False, idx_norm=False, idx_shf=1):
	"""
	Return the the nonzero indices or values if they exists in the array.
	
	Args:
		arr (np.array): 1d numpy array
		ret_idx (bool): boolean to control returning indices or values
		idx_norm (bool): If returning an index, whether to normalize to [0, 1] range
		idx_shf (int ∈ ℤ): value to shift indices by, if found. Only relevant if ret_idx is True.
			This is done to retain the meaning of zero as 'no non-zero value found'.
			The indices correspond to the array after null value removal and shifting by idx_shf.

	Returns:
		None -> The array was all nulls
		0	 -> The non-null values were all zero
		[ℕ]	 -> The first n non-zero values or their indices shifted by idx_shf
	"""
	non_null = filter_null(arr)
	non_zero_ids = np.flatnonzero(non_null)

	if (non_null.size == 0):
		return None
	elif (non_zero_ids.size == 0):
		return 0
	elif (ret_idx):
		indices = non_zero_ids + idx_shf
		return indices / (non_null.size + idx_shf) if (idx_norm) else indices
	else:
		return np.take(non_null, non_zero_ids)


""" ********** PANDAS IO UTILS ********** """
def load_df(fname, dir_path=None, data_format=DF_DATA_FMT, subset=None, dti_freq=None):
	"""
	Read and return the df file in the given directory and
	assume that the file has an index as the first column
	"""
	ext_tuple = FMT_EXTS[data_format]
	fpath = str(add_sep_if_none(dir_path) + fname) if dir_path else fname
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

			if (data_format == 'feather'):
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
	fpath = str(add_sep_if_none(dir_path) + fname) if dir_path else fname
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

def index_intersection(*pd_idx):
	"""
	Return the common intersection of all passed pandas index objects.
	"""
	return reduce(lambda idx, oth: idx.intersection(oth), pd_idx)

def index_split(pd_idx, *ratio):
	"""
	Split an index into multiple sub indexes based on ratios passed in.
	"""
	cuts = get_range_cuts(0, pd_idx.size, list(ratio))
	return tuple(pd_idx[start:end] for start, end in pairwise(cuts))

def pd_common_index_rows(*pd_obj):
	"""
	Take the intersection of pandas object indices and return each object's common indexed rows.
	"""
	common_index = index_intersection(*(obj.index for obj in pd_obj))
	return (obj.loc[common_index] for obj in pd_obj)

def df_count(df):
	return df.count(axis=0)

def df_value_count(df, axis=0):
	"""
	Return value_count for each column of df.
	Setting axis to '1' returns the value count of each column per index (if the range of each
	column is identical, this simulates a vote count of each column for each row/example).
	"""
	return df.apply(lambda ser: ser.value_counts(), axis=axis)

def ser_range_center_clip_tup(ser, range_tuple, inner=0, outer=False, inclusive=False):
	"""
	Return ser with values within threshold range set to inner, and values outside range optionally set
	to threshold.

	Args:
		ser (pd.Series): series to center clip
		thresh ((-float, float)): thresholds for sign binning
		inner (any): what values within the range will be clipped to
		outer (bool): whether or not to clip values outside the range to the threshold
		inclusive (bool): whether the range is inclusive

	Returns:
		thresholded pd.Series
	"""
	out = ser.copy(deep=True)
	out.loc[~pd.isnull(out) & out.between(range_tuple[0], range_tuple[1], inclusive=inclusive)] = inner
	if (outer):
		out.loc[~pd.isnull(out) & out.lt(inner)] = range_tuple[0]
		out.loc[~pd.isnull(out) & out.gt(inner)] = range_tuple[1]
	return out

def ser_range_center_clip_df(ser, range_df, inner=0, outer=False, inclusive=False):
	"""
	Return ser with values within range_df set to inner, and values outside range optionally set
	to threshold.

	Args:
		ser (pd.Series): series to center clip
		range_df (pd.DataFrame): threshold df for center clipping between the range of the first and second columns.
		inner (any): what values within the range will be clipped to
		outer (bool): whether or not to clip values outside the range to the threshold
		inclusive (bool): whether the range is inclusive

	Returns:
		thresholded pd.Series
	"""
	out = ser.copy(deep=True)
	out.loc[pd.isnull(range_df.iloc[:, 0]) | pd.isnull(range_df.iloc[:, 1])] = None
	out.loc[~pd.isnull(out) & out.between(range_df.iloc[:, 0], range_df.iloc[:, 1], inclusive=inclusive)] = inner
	if (outer):
		out.loc[~pd.isnull(out) & out.lt(inner)] = range_df.iloc[:, 0]
		out.loc[~pd.isnull(out) & out.gt(inner)] = range_df.iloc[:, 1]
	return out

def ser_range_center_clip(ser, thresh=None, inner=0, outer=False, inclusive=False):
	"""
	Return ser with values within threshold range set to inner, and values outside range optionally set
	to threshold.

	Args:
		ser (pd.Series): series to center clip
		thresh (pd.DataFrame or float or (float, float)): thresholds for sign binning
			If it is a single threshold, it will be translated to: (-abs(float)/2, abs(float)/2).
			If not, thresholds[0] <= thresholds[1] must be the case.
			where interval[0] < val < interval[1] maps to inner
		inner (any not None): what values within the range will be clipped to
		outer (bool): whether or not to clip values outside the range to the threshold
		inclusive (bool): whether the range is inclusive

	Returns:
		thresholded pd.Series
	"""
	if (is_real_num(thresh) and thresh!=0):
		thresh = (-abs(thresh)/2, abs(thresh)/2) if (is_real_num(thresh)) else thresh

	if (is_seq(thresh)):
		out = ser_range_center_clip_tup(ser, range_tuple=thresh, inner=inner, outer=outer, inclusive=inclusive)
	elif (is_df(thresh)):
		out = ser_range_center_clip_df(ser, range_df=thresh, inner=inner, outer=outer, inclusive=inclusive)
	else:
		out = ser

	return out

"""Datetime"""
def df_dti_index_to_date(df, new_freq=DT_CAL_DAILY_FREQ, new_tz=False):
	"""
	XXX - deprecated
	Convert DataFrame's DatetimeIndex index to solely a date component, set new frequency if specified.
	"""
	index_name = df.index.name
	timezone = new_tz if (new_tz!=False) else df.index.tz
	df.index = pd.DatetimeIndex(df.index.normalize().date).rename(index_name)
	if (new_freq is not None):
		df = df.asfreq(new_freq)
	return df.tz_localize(timezone)

def pd_dti_index_to_date(pd_obj, new_freq=DT_CAL_DAILY_FREQ, new_tz=False):
	"""
	Convert pandas object DatetimeIndex index to solely a date component, set new frequency if specified.
	"""
	index_name = pd_obj.index.name
	timezone = new_tz if (new_tz!=False) else pd_obj.index.tz
	pd_obj.index = pd.DatetimeIndex(pd_obj.index.normalize().date).rename(index_name)
	if (new_freq is not None):
		pd_obj = pd_obj.asfreq(new_freq)
	return pd_obj.tz_localize(timezone)

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

def ser_shift(ser, shift_periods=-1, cast_type=None):
	"""
	Return shifted series, null dropped before and after.
	Used mainly for preparing upshifted time series labels or targets.
	
	Args:
		ser (pd.Series): series to shift
		shift_periods (int): number of periods to shift, by default shifts up by one
		cast_type (type, optional): type to shift to, if None no casting is applied
	"""
	shifted = ser.dropna().shift(periods=shift_periods, freq=None, axis=0).dropna()
	return shifted if (cast_type is None) else shifted.astype(cast_type)

def pd_slot_shift(pd_obj, periods=1, freq=DT_CAL_DAILY_FREQ):
	"""
	Return time series "slot index" shifted over by number of aggregation periods, where
	freq must be larger than the original frequency of the time series.

	This function was needed because pandas shifts according to frequency types, not by
	available "slots" in the index.

	This problem can be solved by custom frequencies, but this method is faster and simpler.

	Args:
		pd_obj (pd.Series or pd.DataFrame): pandas object to slot shift
		periods (int): number of periods (in freq units) to slot shift
		freq (str): shift frequency

	Returns:
		shifted pandas object
	"""
	# Save original index
	idx = pd_obj.dropna(axis=0, how='all').index.copy(deep=True)

	# Downsample to freq using 'last' resample method
	dnsampled = pd_obj.dropna(axis=0, how='all').resample(freq, axis=0, closed='left', label='left').last().dropna(axis=0, how='all')

	# Shift over by periods amount
	shifted = dnsampled.shift(periods=periods, freq=None, axis=0).dropna(axis=0, how='all')

	# Upsample / reindex back to original index
	reindexed = shifted.reindex(index=idx, method='ffill')

	return reindexed

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

def reindex_on_time_mask(reindex_df, time_mask_df, dest_tz_col_name='times'):
	"""
	Convenience function to reindex df by time_mask_df.

	Args:
		reindex_df (pd.DataFrame): DataFrame with index that is a superset (not strict superset) of time_mask_df
		time_mask_df (pd.DataFrame): A table that maps its DatetimeIndex index to the destination timezone

	Returns:
		pd.DataFrame identical to reindex_df, with its index swapped with the 'times' column of time_mask_df
	"""
	new_index = inner_join(reindex_df, time_mask_df)[dest_tz_col_name]
	reindex_df.index = new_index.rename('index')
	return reindex_df

def df_freq_transpose(df, col_freq='hour'):
	"""
	Transpose df by time index attribute.

	Args:
		df (pd.DataFrame): df to transpose
		col_freq (str): columns of result df, must an be attribute of the original df index

	Returns:
		transposed pd.DataFrame 
	"""
	transposed = None
	if (not df.index.empty):
		day_date = df.index.date[0]
		day_cols = getattr(df.index, col_freq)
		transposed = pd.DataFrame(df.values.T, columns=day_cols, index=[day_date])
	return transposed

def gb_transpose(df, agg_freq=DT_CAL_DAILY_FREQ, col_freq='hour'):
	"""
	Convert a series a DataFrame grouped by agg_freq and transposed in each group.

	Args:
		df (pd.DataFrame): df to group-transpose
		agg_freq (str): aggregation frequency

	Returns:
		pd.DataFrame where each aggregation has its rows and columns transposed.
		
		Example:

		def make_example_intraday_df(tz_code='UTC'):
			example_dti = ['2010-01-02 01:00:00+00:00', '2010-01-02 02:00:00+00:00', '2010-01-02 03:00:00+00:00', '2010-01-02 04:00:00+00:00',
							'2010-01-03 01:00:00+00:00', '2010-01-03 02:00:00+00:00', '2010-01-03 03:00:00+00:00', '2010-01-03 04:00:00+00:00',
							'2010-01-04 01:00:00+00:00', '2010-01-04 02:00:00+00:00', '2010-01-04 03:00:00+00:00', '2010-01-04 04:00:00+00:00',
							'2010-01-08 01:00:00+00:00', '2010-01-08 02:00:00+00:00', '2010-01-08 03:00:00+00:00', '2010-01-08 04:00:00+00:00',
							'2010-01-09 01:00:00+00:00',                                                           '2010-01-09 04:00:00+00:00']
			example_vals = [i for i in range(len(example_dti))]
			example = pd.DataFrame(example_vals, index=pd.DatetimeIndex(example_dti), columns=['intraday_vals'])
			example.index.name = 'index'

			for i in example.index:
				if (i.hour==4):
					example.loc[i] = None
			example.loc['2010-01-03 02:00:00+00:00'] = None
			example.loc['2010-01-09 04:00:00+00:00'] = 18

			return example.tz_localize(tz_code)

		Example Input:

			index	                    vals
			2010-01-02 01:00:00+00:00	0.0
			2010-01-02 02:00:00+00:00	1.0
			2010-01-02 03:00:00+00:00	2.0
			2010-01-02 04:00:00+00:00	NaN
			2010-01-03 01:00:00+00:00	4.0
			2010-01-03 02:00:00+00:00	NaN
			2010-01-03 03:00:00+00:00	6.0
			2010-01-03 04:00:00+00:00	NaN
			2010-01-04 01:00:00+00:00	8.0
			2010-01-04 02:00:00+00:00	9.0
			2010-01-04 03:00:00+00:00	10.0
			2010-01-04 04:00:00+00:00	NaN
			2010-01-08 01:00:00+00:00	12.0
			2010-01-08 02:00:00+00:00	13.0
			2010-01-08 03:00:00+00:00	14.0
			2010-01-08 04:00:00+00:00	NaN
			2010-01-09 01:00:00+00:00	16.0
			2010-01-09 04:00:00+00:00	18.0

		Example Output:

			index	    1	  2	    3	  4
			2010-01-02	0.0	  1.0	2.0	  NaN
			2010-01-03	4.0	  NaN	6.0	  NaN
			2010-01-04	8.0	  9.0	10.0  NaN
			2010-01-05	NaN	  NaN	NaN   NaN
			2010-01-06	NaN	  NaN	NaN   NaN
			2010-01-07	NaN	  NaN	NaN   NaN
			2010-01-08	12.0  13.0	14.0  NaN
			2010-01-09	16.0  NaN	NaN	  18.0
	"""
	# Groupby aggfreq
	gbt = df.groupby(pd.Grouper(freq=agg_freq)).apply(df_freq_transpose, col_freq=col_freq)

	# Drop empty columns and MultiIndex
	gbt = gbt.dropna(axis=1, how='all')
	gbt.index = gbt.index.droplevel(level=0)
	gbt.index = pd.to_datetime(gbt.index, yearfirst=True)
	gbt = gbt.asfreq(agg_freq)

	return gbt

"""String"""
def string_df_join_to_ser(df, join_str='_'):
	"""
	Concatenate a dataframe of strings horizontally (across columns) and return the result as a series.
	"""
	return df.apply(lambda ser: join_str.join(ser.tolist()), axis=1)

"""Transforms"""
def pd_abs(pd_obj):
	"""
	Absolute value of all numeric items in pandas DataFrame or Series.
	"""
	return pd_obj.transform(lambda p: p.abs() if p.dtype.kind in 'iufc' else p)

def abs_pd(pd_obj): # XXX - deprecated
	"""
	Absolute value of all numeric items in pandas DataFrame or Series.
	"""
	return pd_obj.transform(lambda p: p.abs() if p.dtype.kind in 'iufc' else p)

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

def pd_is_empty(pd_obj, count_nans=False, how='all', **kwargs):
	if (count_nans):
		return pd_obj.empty
	else:
		return pd_obj.dropna(axis=0, how=how, **kwargs).empty

def is_empty_df(df, count_nans=False, how='all', **kwargs): # XXX - deprecated
	if (count_nans):
		return df.empty
	else:
		return df.dropna(axis=0, how=how, **kwargs).empty

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
ROW_COUNT_THRESH = .1

def filter_cols_below(df, row_count_thresh=ROW_COUNT_THRESH):
	"""
	Return df where the columns whose counts below the row_count_thresh*max column's count are filtered out.
	"""
	df_counts = df.count()
	max_row = max(df_counts)
	low_dropped = df.loc[:, df_counts > (max_row*row_count_thresh)]

	return low_dropped

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

	selected.extend([string for string in qualifier_dict['exact'] if string in str_list])
	selected.extend([string for string in str_list if string.startswith(tuple(qualifier_dict['startswith']))])
	selected.extend([string for string in str_list if string.endswith(tuple(qualifier_dict['endswith']))])
	selected.extend([string for string in str_list if any(re.match(rgx, string) for rgx in qualifier_dict['regex'])])
	if (qualifier_dict['exclude'] is not None):
		exclude_fn = lambda string: string not in get_subset(str_list, qualifier_dict['exclude'])
		selected = filter(exclude_fn, selected)

	return remove_dups_list(selected)

def chained_filter(str_list, qualifier_dict_list):
	"""
	Return subset in str_list that satisfies the list of qualifier_dicts via the procedure of get_subset.

	Args:
		str_list (list(str)): list of strings (ex: a list of column names)
		qualifier_dict_list (list(dict)): list of dictionaries representing a series of filters

	Returns:
		sublist of str_list
	"""
	if (isinstance(qualifier_dict_list, dict)):
		qualifier_dict_list = [qualifier_dict_list]

	return reduce(get_subset, qualifier_dict_list, str_list)


""" ********** DEBUGGING UTILS ********** """
DEFAULT_LOG_LEVEL = logging.INFO

LOG_LEVEL_STR_MAP ={
	'critical': logging.CRITICAL,
	'error': logging.ERROR,
	'warning': logging.WARNING,
	'info': logging.INFO,
	'debug': logging.DEBUG,
	'notset': logging.NOTSET
}

LOG_LEVEL_CHAR_MAP ={
	'c': logging.CRITICAL,
	'e': logging.ERROR,
	'w': logging.WARNING,
	'i': logging.INFO,
	'd': logging.DEBUG,
	'n': logging.NOTSET
}

LOG_LEVEL = ChainMap(LOG_LEVEL_CHAR_MAP, LOG_LEVEL_STR_MAP)

in_debug_mode = lambda: logging.getLogger().isEnabledFor(logging.DEBUG)

def set_loglevel(loglevel=DEFAULT_LOG_LEVEL):
	if (loglevel is None):
		loglevel = DEFAULT_LOG_LEVEL
	elif (isinstance(loglevel, str)):
		loglevel = LOG_LEVEL.get(loglevel, DEFAULT_LOG_LEVEL)

	logging.basicConfig(stream=sys.stdout, level=loglevel)
	logging.getLogger().setLevel(loglevel)


""" ********** PROFILING UTILS ********** """
class benchmark(object):
	"""
	ContextManager that times a selection of code's entry and exit.

	This class was adapted from stackoverflow's user bburns.km code at
		https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python/41408510#41408510
	"""
	def __init__(self, msg, suppress=False, humanized=True):
		self.msg = msg
		self.suppress = suppress
		self.humanized = humanized

	def __enter__(self):
		self.start = default_timer()
		return self

	def __exit__(self, *args):
		elapsed_sec = default_timer() - self.start
		self.delta = timedelta(seconds=elapsed_sec)

		if (not self.suppress):
			delta_str = humanize.naturaltime(self.delta) if (self.humanized) else str(self.delta)
			logging.warning('{msg}: {t}'.format(msg=self.msg, t=delta_str))
