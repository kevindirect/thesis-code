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

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, compose, null_fn, identity_fn, get_custom_biz_freq, window_iter, col_iter, all_equal, is_real_num
from common_util import ser_range_center_clip, pd_slot_shift, concat_map, substr_ad_map, all_equal, first_element, first_letter_concat, arr_nonzero, apply_nz_nn, one_minus
from mutate.common import STANDARD_DAY_LEN
from mutate.tfactory_util import single_row, single_row_map
from mutate.tfactory_util import statistic, difference, fracdiff
from mutate.tfactory_util import window_rank, normalize, symbolize
from mutate.tfactory_util import returnify, expanding_returnify, variable_expanding_returnify


""" ********** JSON-STR-TO-CODE TRANSLATORS ********** """
RUNT_FN_TRANSLATOR = {
	"sr": single_row,
	"srm": single_row_map,
	"stat": statistic,
	"diff": difference,
	"ffd": fracdiff,
	"wr": window_rank,
	"norm": normalize,
	"sym": symbolize,
	"ret": returnify,
	"xret": expanding_returnify,
	"vxret": variable_expanding_returnify
}

RUNT_TYPE_TRANSLATOR = {
	"rut": apply_rut_df,
	"rbt": apply_rbt_df,
	"gbt": apply_gbt_df,
	"gua": apply_gua_df
}

RUNT_NMAP_TRANSLATOR = {
	"cm": concat_map,
	"sami": substr_ad_initial_map,
	None: null_fn
}

RUNT_FREQ_TRANSLATOR = {
	"hourly": DT_HOURLY_FREQ,
	"cal_daily": DT_CAL_DAILY_FREQ,
	"biz_daily": DT_BIZ_DAILY_FREQ,
	DT_HOURLY_FREQ: DT_HOURLY_FREQ,
	DT_CAL_DAILY_FREQ: DT_CAL_DAILY_FREQ,
	DT_BIZ_DAILY_FREQ: DT_BIZ_DAILY_FREQ,
	None: None
}


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
def get_ser_fn(var, ser_fn_str, ser_fns=RUNT_FN_TRANSLATOR):
	ser_fn = list(map(lambda fn: ser_fns.get(fn), ser_fn_str)) if (is_type(ser_fn_str, list)) else [ser_fns.get(ser_fn_str)]
	var = var if (is_type(var, list)) else [var]

	if (len(ser_fn)!=len(var)):
		error_msg = 'number of series functions and variant sets must be equal'
		logging.error(error_msg)
		raise RUNTFormatError(error_msg)
	return [partial(ser_fn[i], var[i]) for i in len(var)]


""" ********** ROW BASED TRANSFORMS ********** """
def apply_rut_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply row unary transform
	"""
	ser_fn = get_ser_fn(var, ser_fn_str)
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
	ser_fn = get_ser_fn(var, ser_fn_str)
	res = df.groupby(pd.Grouper(freq=agg_freq)).transform(ser_transform_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_gua_df(df, var, freq, ser_fn_str, col_fn_str, dna=True):
	"""
	Apply groupby unary aggregation
	"""
	# TODO - TEST
	ser_fn = get_ser_fn(var, ser_fn_str)
	d = {col: df[col].groupby(pd.Grouper(freq=freq)).aggregate(ser_fn[i%len(ser_fn)]) for i, col in enumerate(df.columns)}
	res = DataFrame.from_dict(d)
	#res = df.groupby(pd.Grouper(freq=agg_freq)).agg(ser_agg_fn)
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** COL (NAME MAPPER) FUNCTIONS ********** """
substr_ad_initial_map = partial(substr_ad_map, check_fn=all_equal, accord_fn=first_element, discord_fn=first_letter_concat)


