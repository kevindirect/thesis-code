#                       __
#     _______  ______  / /_
#    / ___/ / / / __ \/ __/
#   / /  / /_/ / / / / /_
#  /_/   \__,_/_/ /_/\__/
# run transforms utilities module.
"""
Kevin Patel
"""
import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import compose, null_fn, identity_fn, get_custom_biz_freq, window_iter, col_iter, all_equal, is_valid, isnt, is_type
from common_util import concat_map, fl_map, window_map, suffix_map
from mutate.common import STANDARD_DAY_LEN
from mutate.tfactory_util import RUNT_FN_MAPPING


""" ********** RUNT EXCEPTION CLASSES ********** """
class RUNTFormatError(Exception):
	"""
	Use this class for formatting/validation errors in Runt Transforms.
	"""
	def __init__(self, message):
		super().__init__(message)

class RUNTComputeError(RuntimeError):
	"""
	Use this class Runt Runtime Errors.
	"""
	def __init__(self, message):
		super().__init__(message)


""" ********** HELPER FNS ********** """
def get_ser_fn(ser_fn_str, var, fn_mapping=RUNT_FN_MAPPING):
	"""
	Convert the string or list of strings to their python function mappings and fix them to a particular set of variables.
	"""
	ser_fn_str = ser_fn_str if (is_type(ser_fn_str, list, tuple)) else (ser_fn_str,)
	var = var if (is_type(var, tuple)) else (var,)

	if (len(ser_fn_str)!=len(var)):
		msg = 'number of ser function strings and variant sets must be equal'
		logging.error(msg)
		raise RUNTFormatError(msg)

	fixed = []
	for fn_str, subvar in zip(ser_fn_str, var):
		fn = fn_mapping.get(fn_str, None)
		if (isnt(fn)):
			msg = 'ser function string \'{}\' does not refer to a function in the fn_mapping'.format(fn_str)
			logging.error(msg)
			raise RUNTFormatError(msg)
		try:
			logging.debug('subvar: {}'.format(subvar))
			fixed.append(fn(**subvar))
		except Exception as e:
			msg = 'error when fixing subvariant to runt transform function'
			logging.error(msg)
			logging.error('trace: {}'.format(e))
			raise RUNTFormatError(msg)

	return fixed


""" ********** ROW BASED TRANSFORMS ********** """
def apply_rut_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply row unary transform
	"""
	ser_fn = get_ser_fn(ser_fn_str, var)
	d = {col: df.loc[:, col].transform(ser_fn[i%len(ser_fn)]) for i, col in enumerate(df.columns)}
	res = pd.DataFrame.from_dict(d)
	if (is_valid(col_fn_str)):
		col_fn = RUNT_NMAP_MAPPING.get(col_fn_str)
		res.columns = col_fn(list(df.columns))
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_rbt_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply row binary transform
	"""
	ser_fn = get_ser_fn(ser_fn_str, var)
	res = pd.DataFrame(index=df.index)
	for i, (col_a, col_b) in enumerate(window_iter(df.columns)):
		res.loc[:, fl_map([col_a, col_b])] = ser_fn[i%len(ser_fn)](df.loc[:, col_a], df.loc[:, col_b])
	if (is_valid(col_fn_str)):
		col_fn = RUNT_NMAP_MAPPING.get(col_fn_str)
		res.columns = col_fn(list(df.columns))
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** GROUP BASED TRANSFORMS ********** """
def apply_gut_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply groupby unary transform
	"""
	ser_fn = get_ser_fn(ser_fn_str, var)
	d = {col: df.loc[:, col].groupby(pd.Grouper(freq=freq)).transform(ser_fn[i%len(ser_fn)]) for i, col in enumerate(df.columns)}
	res = pd.DataFrame.from_dict(d)
	if (is_valid(col_fn_str)):
		col_fn = RUNT_NMAP_MAPPING.get(col_fn_str)
		res.columns = col_fn(list(df.columns))
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_gua_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply groupby unary aggregation
	"""
	ser_fn = get_ser_fn(ser_fn_str, var)
	d = {col: df.loc[:, col].groupby(pd.Grouper(freq=freq)).agg(ser_fn[i%len(ser_fn)]) for i, col in enumerate(df.columns)}
	res = pd.DataFrame.from_dict(d)
	if (is_valid(col_fn_str)):
		col_fn = RUNT_NMAP_MAPPING.get(col_fn_str)
		res.columns = col_fn(list(df.columns))
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_guax_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply groupby unary aggregation, column expanding
	Unlike gua, guax has the potential to add columns to the df
	"""
	ser_fn = get_ser_fn(ser_fn_str, var)
	d = [(df.columns[i%len(df.columns)], df.loc[:, df.columns[i%len(df.columns)]].groupby(pd.Grouper(freq=freq)).agg(fn)) for i, fn in enumerate(ser_fn)]
	res = pd.concat([data[1] for data in d], axis=1, keys=[data[0] for data in d])
	if (is_valid(col_fn_str)):
		col_fn = RUNT_NMAP_MAPPING.get(col_fn_str)
		res.columns = col_fn(list(res.columns))
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** COL (NAME MAPPER) FUNCTIONS ********** """
#substr_ad_initial_map = partial(substr_ad_map, check_fn=all_equal, accord_fn=first_element, discord_fn=first_letter_concat) # DEPRECATED
binary_window_map = partial(window_map, mapper_fn=fl_map, n=2)
ohlca_map = partial(suffix_map, suffixes=['open', 'high', 'low', 'close', 'avgPrice'], modify_unique=False)
close_map = partial(suffix_map, suffixes=['close'], modify_unique=True)


""" ********** RUNT DF MAPPING ********** """
RUNT_TYPE_MAPPING = {
	"rut": apply_rut_df,
	"rbt": apply_rbt_df,
	"gut": apply_gut_df,
	"gua": apply_gua_df,
	"guax": apply_guax_df
}


""" ********** RUNT COL MAPPING ********** """
RUNT_NMAP_MAPPING = {
	"cm": concat_map,
	"bwm": binary_window_map,
	"ohlca": ohlca_map,
	"c": close_map
}
