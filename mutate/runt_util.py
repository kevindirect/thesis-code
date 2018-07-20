# Kevin Patel

import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, get_custom_biz_freq
from mutate.common import STANDARD_DAY_LEN
from mutate.pattern_util import gaussian_breakpoints, uniform_breakpoints, symbolize_value


""" ********** APPLY FUNCTIONS ********** """
def apply_rt_df(df, ser_transform_fn, freq=None): # regular transform
	return df.transform(ser_transform_fn)

def apply_gbt_df(df, ser_transform_fn, agg_freq): # groupby transform
	return df.groupby(pd.Grouper(freq=agg_freq)).transform(ser_transform_fn)

def apply_gbf_df(df, ser_filter_fn, agg_freq): # groupby filter
	return df.groupby(pd.Grouper(freq=agg_freq)).filter(ser_filter_fn)


""" ********** TRANSFORMS ********** """
def difference(num_periods):
	return lambda ser: ser.diff(periods=num_periods)

def moving_average(num_periods):
	return lambda ser: ser.rolling(window=num_periods, min_periods=num_periods).mean()

NORM_FUN_MAP = {
	'dzn': lambda ser: (ser-ser.mean()) / ser.std(),
	'dmx': lambda ser: 2 * ((ser-ser.min()) / (ser.max()-ser.min())) - 1
}

def normalize(norm_type):
	return lambda ser: ser.transform(NORM_FUN_MAP[norm_type])

SYM_MAP = {
	"gau": gaussian_breakpoints,
	"uni": uniform_breakpoints
}

def symbolize(sym_type, num_sym, numeric_symbols=True):
	"""
	Return symbolization encoder.
	Does not perform paa or any other subseries downsampling/aggregation.

	Args:
		sym_type (str): determines type of breakpoint used
		num_sym (int): alphabet size
		numeric_symbols (boolean): numeric or non-numeric symbols

	Return:
		symbol encoder function
	"""
	breakpoint_dict = SYM_MAP[sym_type]
	breakpoints = breakpoint_dict[num_sym]
	symbols = get_sym_list(breakpoints, numeric_symbols=numeric_symbols)
	encoder = partial(symbolize_value, breakpoints=breakpoints, symbols=symbols)

	logging.debug('breakpoints: ' +str(breakpoints))
	logging.debug('symbols: ' +str(symbols))

	return encoder


""" ********** FILTERS ********** """
def single_row_filter(which_row):
	if (which_row == 'f'):
		return lambda ser: ser.first()
	elif (which_row == 'l'):
		return lambda ser: ser.last()
	elif (isinstance(which_row, int)):
		return lambda ser: ser.nth(which_row)


""" ********** JSON-STR-TO-CODE TRANSLATORS ********** """
RUNT_FN_TRANSLATOR = {
	"diff": difference,
	"ma": moving_average,
	"norm": normalize,
	"sym": symbolize,
	"srf": single_row_filter,
}

RUNT_TYPE_TRANSLATOR = {
	"rt": apply_rt_df,
	"gbt": apply_gbt_df,
	"gbf": apply_gbf_df
}

RUNT_FREQ_TRANSLATOR = {
	"hourly": DT_HOURLY_FREQ,
	"cal_daily": DT_CAL_DAILY_FREQ,
	"biz_daily": DT_BIZ_DAILY_FREQ,
	None: None
}