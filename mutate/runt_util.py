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

from common_util import compose, null_fn, identity_fn, get_custom_biz_freq, window_iter, col_iter, all_equal, is_real_num, is_type
from common_util import ser_range_center_clip, pd_slot_shift, concat_map, substr_ad_map, all_equal, first_element, first_letter_concat, arr_nonzero, apply_nz_nn, one_minus
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
	ser_fn = tuple(map(lambda fn: fn_mapping.get(fn), ser_fn_str)) if (is_type(ser_fn_str, list)) else tuple(fn_mapping.get(ser_fn_str))
	var = var if (is_type(var, tuple)) else tuple(var)

	if (len(ser_fn)!=len(var)):
		msg = 'number of ser functions and variant sets must be equal'
		logging.error(msg)
		raise RUNTFormatError(msg)

	fixed = []
	for fn, subvar in zip(ser_fn, var):
		try:
			fixed.append(fn(subvar))
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
	d = {col: df[col].transform(ser_fn[i%len(ser_fn)]) for i, col in enumerate(df.columns)}
	res = pd.DataFrame(d, index=df.index)
	# TODO
	#res = df.transform(ser_transform_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_rbt_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply row binary transform
	"""
	res = pd.DataFrame(index=df.index)
	for col_a, col_b in window_iter(df.columns):
		res.loc[:, name_map_fn(col_a, col_b)] = binary_apply_fn(df[col_a], df[col_b])
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** GROUP BASED TRANSFORMS ********** """
def apply_gut_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply groupby unary transform
	"""
	# TODO
	ser_fn = get_ser_fn(ser_fn_str, var)
	res = df.groupby(pd.Grouper(freq=freq)).transform(ser_transform_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_gua_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply groupby unary aggregation
	"""
	# TODO - TEST
	ser_fn = get_ser_fn(ser_fn_str, var)
	d = {col: df.loc[:, col].groupby(pd.Grouper(freq=freq)).agg(ser_fn[i%len(ser_fn)]) for i, col in enumerate(df.columns)}
	res = DataFrame.from_dict(d)
	#res = df.groupby(pd.Grouper(freq=agg_freq)).agg(ser_agg_fn)
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** COL (NAME MAPPER) FUNCTIONS ********** """
substr_ad_initial_map = partial(substr_ad_map, check_fn=all_equal, accord_fn=first_element, discord_fn=first_letter_concat)


""" ********** RUNT DF MAPPING ********** """
RUNT_TYPE_MAPPING = {
	"rut": apply_rut_df,
	"rbt": apply_rbt_df,
	"gut": apply_gut_df,
	"gua": apply_gua_df
}


""" ********** RUNT COL MAPPING ********** """
RUNT_NMAP_MAPPING = {
	"cm": concat_map,
	"sami": substr_ad_initial_map,
	None: null_fn
}
