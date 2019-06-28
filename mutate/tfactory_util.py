#     __  ____           __
#    / /_/ __/___ ______/ /_____  _______  __
#   / __/ /_/ __ `/ ___/ __/ __ \/ ___/ / / /
#  / /_/ __/ /_/ / /__/ /_/ /_/ / /  / /_/ /
#  \__/_/  \__,_/\___/\__/\____/_/   \__, /
#                                   /____/
# transforms factory utilities module.
# second-order data transform functions.
"""
Kevin Patel
"""
import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, DT_BIZ_DAILY_FREQ, compose, null_fn, identity_fn, get_custom_biz_freq, window_iter, col_iter, all_equal, is_real_num
from common_util import ser_range_center_clip, pd_slot_shift, concat_map, substr_ad_map, all_equal, first_element, first_letter_concat, arr_nonzero, apply_nz_nn, one_minus, is_valid, isnt
from mutate.common import STANDARD_DAY_LEN
from mutate.pattern_util import gaussian_breakpoints, uniform_breakpoints, get_sym_list, symbolize_value
from mutate.fracdiff_util import get_weights


""" ********** sr **********"""
def first_nonzero(ser, ret_idx=False, idx_norm=False):
	idx = arr_nonzero(ser.values, ret_idx=ret_idx, idx_norm=idx_norm, idx_shf=1)
	return idx if (not isinstance(idx, np.ndarray) and (isnt(idx) or idx==0)) else idx.item(0)

ROW_IDX_SELECTOR_MAPPING = {
	0: (lambda ser: ser.first_valid_index()),
	-1: (lambda ser: ser.last_valid_index()),
	'h': (lambda ser: ser.idxmax(skipna=True)),
	'l': (lambda ser: ser.idxmin(skipna=True)),
	'fnz': partial(first_nonzero, ret_idx=True, idx_norm=False),
	'fnz_idxscore': partial(first_nonzero, ret_idx=True, idx_norm=True)	# Returns index as normalized score
}
ROW_VAL_SELECTOR_MAPPING = {
	0: (lambda ser: ser.loc[ROW_IDX_SELECTOR_MAPPING.get(0)(ser)]),
	-1: (lambda ser: ser.loc[ROW_IDX_SELECTOR_MAPPING.get(-1)(ser)]),
	'h': (lambda ser: ser.loc[ROW_IDX_SELECTOR_MAPPING.get('h')(ser)]),
	'l': (lambda ser: ser.loc[ROW_IDX_SELECTOR_MAPPING.get('l')(ser)]),
	'fnz': partial(first_nonzero, ret_idx=False, idx_norm=False)
}
def single_row(val, flt):
	"""
	Constructs function that returns index or value for selected row.
	"""
	return {
		bool(not val): ROW_IDX_SELECTOR_MAPPING.get(flt, None),
		val: ROW_VAL_SELECTOR_MAPPING.get(flt, None)
	}.get(True)


""" ********** srm **********"""
MAP_FN_MAPPING = {
	'nz_inv': apply_nz_nn(one_minus),
	'sgn': np.sign
}
def single_row_map(val, flt, map_fn):
	return compose(single_row(val, flt), MAP_FN_MAPPING.get(map_fn))


""" ********** stat **********"""
STAT_FN_MAPPING = {
	'avg': pd.Series.mean,
	'std': pd.Series.std,
	'max': pd.Series.max,
	'min': pd.Series.min
}
def statistic(stat_type):
	return STAT_FN_MAPPING.get(stat_type)


""" ********** aggstat **********"""
def aggregated_statistic(stat_type, abs_val=True, agg_freq=DT_CAL_DAILY_FREQ):
	stat_fn = statistic(stat_type)

	def fn(ser):
		if (abs_val):
			out = ser.abs().groupby(pd.Grouper(freq=agg_freq)).transform(stat_fn)
		else:
			out = ser.groupby(pd.Grouper(freq=agg_freq)).transform(stat_fn)
		return out

	return fn


""" ********** diff **********"""
def difference(num_periods):
	def fn(ser):
		return ser.diff(periods=num_periods)
	return fn


""" ********** ffd **********"""
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


""" ********** wr **********"""
RANK_FN_MAPPING = {
	'zn': (lambda ser: (ser.iloc[-1]-ser.mean()) / ser.std()),
	'mx': (lambda ser: 2 * ((ser.iloc[-1]-ser.min()) / (ser.max()-ser.min())) - 1),
	'od': (lambda ser, normalize=True: ser.rank(numeric_only=True, ascending=True, pct=normalize).iloc[-1]),
	'pt': (lambda ser: (ser.iloc[-1]-ser.min()) / (ser.max()-ser.min()))
}
def window_rank(rank_type, num_periods):
	rank_fn = RANK_FN_MAPPING.get(rank_type)

	def fn(ser):
		win = ser.expanding(min_periods=1) if (num_periods==-1) else ser.rolling(window=num_periods, min_periods=1)
		return win.apply(rank_fn)

	return fn


""" ********** norm **********"""
def normalize(norm_type):
	norm_fn = RANK_FN_MAPPING.get(norm_type)

	def fn(ser):
		return ser.transform(norm_fn)

	return fn


""" ********** sym **********"""
SYM_MAPPING = {
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
	breakpoints = SYM_MAPPING[sym_type][num_sym]
	symbols = get_sym_list(breakpoints, numeric_symbols=numeric_symbols)
	encoder = partial(symbolize_value, breakpoints=breakpoints, symbols=symbols)

	def fn(ser):
		return ser.transform(encoder)

	logging.debug('breakpoints: {}'.format(str(breakpoints)))
	logging.debug('symbols: {}'.format(str(symbols)))

	return fn


""" ********** ret **********"""
RETURN_FN_MAPPING = {
	"spread": (lambda slow_ser, fast_ser: fast_ser - slow_ser),
	"ret": (lambda slow_ser, fast_ser: (fast_ser / slow_ser) - 1),
	"logret": (lambda slow_ser, fast_ser: np.log(fast_ser / slow_ser))
}
def returnify(ret_type, thresh=None, clip=False):
	"""
	Spread, regular return, or log return.
	"""
	ret_fn = RETURN_FN_MAPPING.get(ret_type)

	def fn(slow_ser, fast_ser):
		ret = ret_fn(slow_ser, fast_ser)
		if (is_valid(thresh)):
			ret = ser_range_center_clip(ret, thresh, inner=SIDEWAYS, outer=clip, inclusive=False)
		return ret
	return fn


""" ********** xret **********"""
def expanding_returnify(ret_type, thresh=None, clip=False, agg_freq=DT_CAL_DAILY_FREQ):
	"""
	Expanding spread, regular return, or log return.
	"""
	ret_fn = returnify(ret_type, thresh=thresh, clip=clip)

	def fn(slow_ser, fast_ser):
		first_slow = slow_ser.dropna().groupby(pd.Grouper(freq=agg_freq)).transform(single_row(True, 0))
		return ret_fn(first_slow, fast_ser.dropna())

	return fn


""" ********** vxret **********"""
def variable_expanding_returnify(ret_type, stat_type, clip=False, thresh_scalar=1, agg_freq=DT_CAL_DAILY_FREQ):
	"""
	Expanding return thresholded on past period statistic.
	"""
	ret_fn = expanding_returnify(ret_type, thresh=None, clip=False, agg_freq=agg_freq)
	stat_fn = aggregated_statistic(stat_type, abs_val=True, agg_freq=agg_freq)

	def fn(slow_ser, fast_ser):
		ret = ret_fn(slow_ser, fast_ser)
		stat = stat_fn(ret) * thresh_scalar
		thresh = pd_slot_shift(pd.DataFrame({0:-stat, 1: stat}, index=ret.index), periods=1, freq=agg_freq)
		idx = ret.index & thresh.index
		clipped = ser_range_center_clip(ret.loc[idx], thresh.loc[idx], inner=SIDEWAYS, outer=clip, inclusive=False)
		return clipped

	return fn


""" ********** RUNT SER FN MAPPING ********** """
RUNT_FN_MAPPING = {
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

