"""
Kevin Patel
"""

import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, null_fn, get_custom_biz_freq, window_iter, col_iter, all_equal, is_real_num
from common_util import concat_map, substr_ad_map, all_equal, first_element, first_letter_concat
from mutate.common import STANDARD_DAY_LEN
from mutate.pattern_util import gaussian_breakpoints, uniform_breakpoints, get_sym_list, symbolize_value
from mutate.fracdiff import get_weights
from mutate.label_util import UP, DOWN, SIDEWAYS, fastbreak_eod_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct

""" ********** APPLY FUNCTIONS ********** """
def apply_rt_df(df, ser_transform_fn, freq, name_map_fn, dna=True):	# regular transform
	res = df.transform(ser_transform_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_gbt_df(df, ser_transform_fn, agg_freq, name_map_fn, dna=True):	# groupby transform
	res = df.groupby(pd.Grouper(freq=agg_freq)).transform(ser_transform_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_gagg_df(df, ser_agg_fn, agg_freq, name_map_fn, dna=True):		# groupby aggregation
	res = df.groupby(pd.Grouper(freq=agg_freq)).agg(ser_agg_fn)
	return res.dropna(axis=0, how='all') if (dna) else res

def apply_btw_df(df, binary_apply_fn, freq, name_map_fn, dna=True):	# binary transform window function
	res = pd.DataFrame(index=df.index)
	for col_a, col_b in window_iter(df.columns):
		res[name_map_fn(col_a, col_b)] = binary_apply_fn(df[col_a], df[col_b])
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** NAME MAPPER FUNCTIONS ********** """
substr_ad_initial_map = partial(substr_ad_map, check_fn=all_equal, accord_fn=first_element, discord_fn=first_letter_concat)


""" ********** TRANSFORMS ********** """
def difference(num_periods):
	return lambda ser: ser.diff(periods=num_periods)

RETURN_FUN_MAP = {
	"spread": (lambda slow_ser, fast_ser: fast_ser - slow_ser),
	"return": (lambda slow_ser, fast_ser: (fast_ser / slow_ser) - 1),
	"logret": (lambda slow_ser, fast_ser: np.log(fast_ser / slow_ser))
}

def returnize(ret_type):
	"""
	Spread, regular return, or log return.
	"""
	return RETURN_FUN_MAP[ret_type]

def expanding_returnize(ret_type):
	"""
	Spread, regular return, or log return.
	"""
	def ret(slow_ser, fast_ser):
		first_slow = gb['slow'].transform(pd.Series.first, org_freq)
		derived[_cname('xwhole')] = RETURN_FUN_MAP[ret_type](fast_ser, first_slow)
	return RETURN_FUN_MAP[ret_type]

def expanding_fracdiff(d, size, thresh):
	"""
	Expanding Window Fractional Differencing
	XXX - Implement
	"""
	pass
# 	def expanding_fracdiff(ser):
# 		_ser = ser.dropna()
# 		weights = get_weights(d, size=_ser.size, thresh=thresh)
# 		# Determine initial calcs to be skipped based on weight-loss threshold
# 		weight_changes = np.cumsum(abs(weights))
# 		weight_changes /= weight_changes[-1]
# 		skip = weight_changes[weight_changes > thresh].shape[0]

# 	return lambda ser: ser.transform(dot_weights_fn)

def fixed_fracdiff(d, size, thresh):
	"""
	Fixed Window Fractional Differencing
	"""
	weights = get_weights(d, size=size, thresh=thresh)
	dot_weights_fn = lambda ser: ser.dot(weights)
	return lambda ser: ser.dropna().rolling(window=len(weights), min_periods=len(weights)).apply(dot_weights_fn)

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

	return lambda ser: ser.transform(encoder)


""" ********** LABEL EXTRACTION ********** """
def threshold_sign_discretizer(thresh):
	"""
	Return a function that returns a sign discretizer

	Args:
		thresh (float or (float, float)): threshold or thresholds for sign binning
			If it is a single threshold, it will be translated to: (-abs(float)/2, abs(float)/2).
			If not, thresholds[0] <= thresholds[0] must be the case.
			where:	val <= interval[0] maps to DOWN
					val >= interval[1] maps to UP
					interval[0] < val < interval[1] maps to SIDEWAYS
	"""
	def thresh_sign_binner(ser):
		threshes = (-abs(thresh)/2, abs(thresh)/2) if (is_real_num(thresh)) else thresh
		thresholded = ser.copy(deep=True)

		thresholded[thresholded <= threshes[0]] = DOWN
		thresholded[thresholded >= threshes[1]] = UP
		thresholded[~pd.isnull(thresholded) & ~thresholded.isin((UP, DOWN))] = SIDEWAYS

		return thresholded.dropna()

	return thresh_sign_binner

FCT_MASK_MAP = {
	'fbeod': fastbreak_eod_fct,
	'fb': fastbreak_fct,
	'conf': confidence_fct,
	'fbconf': fastbreak_confidence_fct,
	'vel': partial(fastbreak_fct, velocity=True),
	'mag': partial(confidence_fct, magnitude=True),
	'mom': partial(fastbreak_confidence_fct, momentum=True),
}

def mask_return(label_type, mask_type):
	"""
	Return a function that takes a label_df and returns labels or targets according to a masking type.

	Args:
		label_type ('label' | 'target'): integer label or real number target
		mask_type (str): Specific mask type to extract from label_df

	Return:
		labellizer function
	"""
	pass


def mask_labellizer(label_type, mask_type):
	"""
	Return a function that takes a label_df and returns labels or targets according to a masking type.

	Args:
		label_type ('label' | 'target'): integer label or real number target
		mask_type (str): Specific mask type to extract from label_df

	Return:
		labellizer function
	"""
	pass


""" ********** FILTERS ********** """
def single_row_filter(specifier):
	if (specifier == 'f'):
		return lambda ser: ser.loc[ser.first_valid_index()] if (ser.first_valid_index() is not None) else None
	elif (specifier == 'l'):
		return lambda ser: ser.loc[ser.last_valid_index()] if (ser.last_valid_index() is not None) else None
	elif (isinstance(specifier, int)):
		return lambda ser: ser.nth(specifier)


""" ********** JSON-STR-TO-CODE TRANSLATORS ********** """
RUNT_FN_TRANSLATOR = {
	"diff": difference,
	"ret": returnize,
	"efracdiff": expanding_fracdiff,
	"ffracdiff": fixed_fracdiff,
	"ma": moving_average,
	"norm": normalize,
	"sym": symbolize,
	"srf": single_row_filter,
	"threshsign": threshold_sign_discretizer
}

RUNT_TYPE_TRANSLATOR = {
	"rt": apply_rt_df,
	"gbt": apply_gbt_df,
	"gagg": apply_gagg_df,
	"btw": apply_btw_df
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
	None: None
}
