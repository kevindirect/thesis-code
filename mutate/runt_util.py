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
from mutate.tfactory_util import *


""" ********** APPLY FUNCTIONS ********** """
def apply_rt_df(df, var, freq, ser_transform_fn_str, name_map_fn_str, dna=True):	# regular transform
	res = df.transform(ser_transform_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_gbt_df(df, ser_transform_fn, agg_freq, name_map_fn, dna=True):	# groupby transform
	res = df.groupby(pd.Grouper(freq=agg_freq)).transform(ser_transform_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_gagg_df(df, ser_agg_fn, agg_freq, name_map_fn, dna=True):		# groupby aggregation
	res = df.groupby(pd.Grouper(freq=agg_freq)).agg(ser_agg_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_btw_df(df, binary_apply_fn, freq, name_map_fn, dna=True):	# binary transform sliding window
	res = pd.DataFrame(index=df.index)
	for col_a, col_b in window_iter(df.columns):
		res.loc[:, name_map_fn(col_a, col_b)] = binary_apply_fn(df[col_a], df[col_b])
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** NAME MAPPER FUNCTIONS ********** """
substr_ad_initial_map = partial(substr_ad_map, check_fn=all_equal, accord_fn=first_element, discord_fn=first_letter_concat)


""" ********** JSON-STR-TO-CODE TRANSLATORS ********** """
RUNT_FN_TRANSLATOR = {
	"diff": difference,
	"ret": returnify,
	"retx": expanding_returnify,
	"vretx": variable_expanding_returnify,
	"ffd": fracdiff,
	"wr": window_rank,
	"norm": normalize,
	"sym": symbolize,
	"srf": single_row_filter,
	"sgn": signify
}

RUNT_TYPE_TRANSLATOR = {
	"rut": apply_rt_df,
	"gbt": apply_gbt_df,
	"gua": apply_gagg_df,
	"rbt": apply_btw_df
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


""" ********** RUNT Exceptions ********** """
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

