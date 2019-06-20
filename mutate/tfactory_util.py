#     __  ____           __
#    / /_/ __/___ ______/ /_____  _______  __
#   / __/ /_/ __ `/ ___/ __/ __ \/ ___/ / / /
#  / /_/ __/ /_/ / /__/ /_/ /_/ / /  / /_/ /
#  \__/_/  \__,_/\___/\__/\____/_/   \__, /
#                                   /____/
# transforms factory utilities module.
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
from common_util import ser_range_center_clip, pd_slot_shift, concat_map, substr_ad_map, all_equal, first_element, first_letter_concat, arr_nonzero, apply_nz_nn, one_minus, is_valid, isnt
from mutate.common import STANDARD_DAY_LEN
from mutate.pattern_util import gaussian_breakpoints, uniform_breakpoints, get_sym_list, symbolize_value
from mutate.fracdiff_util import get_weights


""" ********** TRANSFORMS ********** """
STAT_FN_MAP = {
	'avg': pd.Series.mean,
	'std': pd.Series.std,
	'max': pd.Series.max,
	'min': pd.Series.min
}

def statistic(stat_type, abs_val=True, agg_freq='cal_daily'):
	agg_freq = RUNT_FREQ_TRANSLATOR.get(agg_freq)
	stat_fn = STAT_FN_MAP.get(stat_type)

	def fn(ser):
		if (abs_val):
			out = ser.abs().groupby(pd.Grouper(freq=agg_freq)).transform(stat_fn)
		else:
			out = ser.groupby(pd.Grouper(freq=agg_freq)).transform(stat_fn)
		return out

	return fn

def difference(num_periods):
	def fn(ser):
		return ser.diff(periods=num_periods)
	return fn

RANK_FN_MAP = {
	'zn': (lambda ser: (ser.iloc[-1]-ser.mean()) / ser.std()),
	'mx': (lambda ser: 2 * ((ser.iloc[-1]-ser.min()) / (ser.max()-ser.min())) - 1),
	'od': (lambda ser, normalize=True: ser.rank(numeric_only=True, ascending=True, pct=normalize).iloc[-1]),
	'pt': (lambda ser: (ser.iloc[-1]-ser.min()) / (ser.max()-ser.min()))
}

def window_rank(rank_type, num_periods):
	rank_fn = RANK_FN_MAP.get(rank_type)

	def fn(ser):
		win = ser.expanding(min_periods=1) if (num_periods==-1) else ser.rolling(window=num_periods, min_periods=1)
		return win.apply(rank_fn)

	return fn

def normalize(norm_type):
	norm_fn = RANK_FN_MAP.get(norm_type)

	def fn(ser):
		return ser.transform(norm_fn)

	return fn

RETURN_FN_MAP = {
	"spread": (lambda slow_ser, fast_ser: fast_ser - slow_ser),
	"ret": (lambda slow_ser, fast_ser: (fast_ser / slow_ser) - 1),
	"logret": (lambda slow_ser, fast_ser: np.log(fast_ser / slow_ser))
}

def returnify(ret_type, thresh=None, clip=False):
	"""
	Spread, regular return, or log return.
	"""
	ret_fn = RETURN_FN_MAP.get(ret_type)

	def fn(slow_ser, fast_ser):
		ret = ret_fn(slow_ser, fast_ser)
		if (is_valid(thresh)):
			ret = ser_range_center_clip(ret, thresh, inner=SIDEWAYS, outer=clip, inclusive=False)
		return ret

	return fn

def expanding_returnify(ret_type, thresh=None, clip=False, agg_freq='cal_daily'):
	"""
	Expanding spread, regular return, or log return.
	"""
	ret_fn = returnify(ret_type, thresh=thresh, clip=clip)
	agg_freq = RUNT_FREQ_TRANSLATOR.get(agg_freq)

	def fn(slow_ser, fast_ser):
		first_slow = slow_ser.dropna().groupby(pd.Grouper(freq=agg_freq)).transform(single_row_filter('f'))
		return ret_fn(first_slow, fast_ser.dropna())

	return fn

def variable_expanding_returnify(ret_type, stat_type, clip=False, thresh_scalar=1, agg_freq='cal_daily'):
	"""
	Expanding return thresholded on past period statistic.
	"""
	ret_fn = expanding_returnify(ret_type, thresh=None, clip=False, agg_freq=agg_freq)
	stat_fn = statistic(stat_type, abs_val=True, agg_freq=agg_freq)
	agg_freq = RUNT_FREQ_TRANSLATOR.get(agg_freq)

	def fn(slow_ser, fast_ser):
		ret = ret_fn(slow_ser, fast_ser)
		stat = stat_fn(ret) * thresh_scalar
		thresh = pd_slot_shift(pd.DataFrame({0:-stat, 1: stat}, index=ret.index), periods=1, freq=agg_freq)
		idx = ret.index & thresh.index
		clipped = ser_range_center_clip(ret.loc[idx], thresh.loc[idx], inner=SIDEWAYS, outer=clip, inclusive=False)
		return clipped

	return fn

def sign(val):
	if (val is None):
		return None
	else:
		return {
			val > 0: 1,
			val < 0: -1,
			val == 0: 0
		}.get(True)

def signify():
	"""
	Map a series through the sign function.
	"""
	def sgn(ser):
		return ser.copy().map(sign, na_action='ignore')
	return sgn

def binary_sign_difference_count():
	"""
	Return the difference in positive versus negative signs divided by all.
	"""
	pass

def fracdiff(d, thresh=None, size=None):
	"""
	Fractional Differencing
	Will be Expanding, Fixed Window, or Lower Bound of the two depending on the parameters passed in
	"""
	weights = get_weights(d, size=size, thresh=thresh)
	dot_weights_fn = lambda ser: ser.dot(weights)

	def fn(ser):
		return ser.dropna().rolling(window=len(weights), min_periods=len(weights)).apply(dot_weights_fn)
	return fn

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
	breakpoints = SYM_MAP[sym_type][num_sym]
	symbols = get_sym_list(breakpoints, numeric_symbols=numeric_symbols)
	encoder = partial(symbolize_value, breakpoints=breakpoints, symbols=symbols)

	def fn(ser):
		return ser.transform(encoder)

	logging.debug('breakpoints: {}'.format(str(breakpoints)))
	logging.debug('symbols: {}'.format(str(symbols)))

	return fn


""" ********** FILTERS ********** """
DEFAULT_RET_IDX = False 	# Default to return value instead of index
DEFAULT_IDX_NORM = False	# Default to return index value instead of normalized score
DEFAULT_FNZ_IDX_SHF = 1		# Default to return index shifted by +1

def row_first(ser, ret_idx=DEFAULT_RET_IDX):
	idx = ser.first_valid_index()

	if (isnt(idx)):
		return None
	elif (or ret_idx):
		return idx
	else:
		return ser.loc[idx]

def row_last(ser, ret_idx=DEFAULT_RET_IDX):
	idx = ser.last_valid_index()

	if (isnt(idx)):
		return None
	elif (or ret_idx):
		return idx
	else:
		return ser.loc[idx]

def first_nonzero(ser, ret_idx=DEFAULT_RET_IDX, idx_norm=DEFAULT_IDX_NORM):
	idx = arr_nonzero(ser.values, ret_idx=ret_idx, idx_norm=idx_norm, idx_shf=DEFAULT_FNZ_IDX_SHF)

	if (not isinstance(idx, np.ndarray) and (isnt(idx) or idx==0)):
		return idx
	else:
		return idx.item(0)

def nth_only(ser, ret_idx=DEFAULT_RET_IDX):
	if (ret_idx):
		return ser.nth(specifier).index
	else:
		return ser.nth(specifier)

SRF_SPECIFIER = {
	0: first_only,
	-1: last_only,
	'h': None,
	'l': None,
	'fnz': first_nonzero,
	'fnzi_normed': partial(first_nonzero, ret_idx=True, idx_norm=True),
	None: nth_only
}

SRF_MAP_FN = {
	'nz_inv': apply_nz_nn(one_minus),
	'sgn': sign,
	None: identity_fn
}

def single_row_filter(specifier, map_fn=None):
	"""
	Return a function that filters a series down to one row and is mapped through
	a mapping function.

	Args:
		specifier (str): type of filter to use
		map_fn (str): mapping function to use

	Return:
		filter map function
	"""
	flt = SRF_SPECIFIER.get(specifier, nth_only)
	mp = SRF_MAP_FN.get(map_fn, identity_fn)		# TODO - fix
	fn = compose(flt, mp)
	return fn

