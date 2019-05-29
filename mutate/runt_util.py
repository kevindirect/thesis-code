"""
Kevin Patel
"""
import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd
from numba import jit

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, compose, null_fn, identity_fn, get_custom_biz_freq, window_iter, col_iter, all_equal, is_real_num
from common_util import ser_range_center_clip, pd_slot_shift, concat_map, substr_ad_map, all_equal, first_element, first_letter_concat, arr_nonzero, apply_nz_nn, one_minus
from mutate.common import STANDARD_DAY_LEN
from mutate.pattern_util import gaussian_breakpoints, uniform_breakpoints, get_sym_list, symbolize_value
from mutate.fracdiff_util import get_weights


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

def apply_btw_df(df, binary_apply_fn, freq, name_map_fn, dna=True):	# binary transform sliding window
	res = pd.DataFrame(index=df.index)
	for col_a, col_b in window_iter(df.columns):
		res.loc[:, name_map_fn(col_a, col_b)] = binary_apply_fn(df[col_a], df[col_b])
	return res.dropna(axis=0, how='all') if (dna) else res


""" ********** NAME MAPPER FUNCTIONS ********** """
substr_ad_initial_map = partial(substr_ad_map, check_fn=all_equal, accord_fn=first_element, discord_fn=first_letter_concat)


""" ********** TRANSFORMS ********** """
STAT_FN_MAP = {
	'avg': pd.Series.mean,
	'std': pd.Series.std,
	'max': pd.Series.max,
	'min': pd.Series.min
}

def statistic(stat_type, abs_val=True, agg_freq='cal_daily'):
	agg_freq = RUNT_FREQ_TRANSLATOR[agg_freq]
	stat_fn = STAT_FN_MAP[stat_type]

	def stat(ser):
		if (abs_val):
			out = ser.abs().groupby(pd.Grouper(freq=agg_freq)).transform(stat_fn)
		else:
			out = ser.groupby(pd.Grouper(freq=agg_freq)).transform(stat_fn)

		return out

	return stat

def difference(num_periods):
	return lambda ser: ser.diff(periods=num_periods)

zscore_rank = lambda ser: (ser.iloc[-1]-ser.mean()) / ser.std()
min_max_rank = lambda ser: 2 * ((ser.iloc[-1]-ser.min()) / (ser.max()-ser.min())) - 1
ordinal_rank = lambda ser, normalize=True: ser.rank(numeric_only=True, ascending=True, pct=normalize).iloc[-1]
percentile_rank = lambda ser: (ser.iloc[-1]-ser.min()) / (ser.max()-ser.min())

def simple_moving_window(stat_type, num_periods):
	fn = {
		'avg': lambda ser: ser.rolling(window=num_periods, min_periods=1).avg(),
		'std': lambda ser: ser.rolling(window=num_periods, min_periods=1).std(),
		'var': lambda ser: ser.rolling(window=num_periods, min_periods=1).var(),
		'zsc': lambda ser: ser.rolling(window=num_periods, min_periods=1).apply(zscore_rank),
		'mmx': lambda ser: ser.rolling(window=num_periods, min_periods=1).apply(min_max_rank),
		'pct': lambda ser: ser.rolling(window=num_periods, min_periods=1).apply(percentile_rank),
		'ord': lambda ser: ser.rolling(window=num_periods, min_periods=1).apply(ordinal_rank),
		'min': lambda ser: ser.rolling(window=num_periods, min_periods=1).min(),
		'max': lambda ser: ser.rolling(window=num_periods, min_periods=1).max(),
		'skw': lambda ser: ser.rolling(window=num_periods, min_periods=1).skew(),
		'krt': lambda ser: ser.rolling(window=num_periods, min_periods=1).kurt()
		,
	}.get(stat_type)
	return fn

def expanding_window(stat_type):
	fn = {
		'avg': lambda ser: ser.expanding(min_periods=1).avg(),
		'std': lambda ser: ser.expanding(min_periods=1).avg(),
		'var': lambda ser: ser.expanding(min_periods=1).avg(),
		'zsc': lambda ser: ser.expanding(min_periods=1).apply(zscore_rank),
		'mmx': lambda ser: ser.expanding(min_periods=1).apply(min_max_rank),
		'pct': lambda ser: ser.expanding(min_periods=1).apply(percentile_rank),
		'ord': lambda ser: ser.expanding(min_periods=1).apply(ordinal_rank)
	}.get(stat_type)
	return fn

def exponential_moving_window(stat_type, num_periods):
	fn = {
		'avg': lambda ser: ser.ewm(span=num_periods, min_periods=1).avg(),
		'std': lambda ser: ser.ewm(span=num_periods, min_periods=1).std(),
		'var': lambda ser: ser.ewm(span=num_periods, min_periods=1).var()
	}.get(stat_type)
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
	ret_fn = RETURN_FN_MAP[ret_type]

	if (thresh is None):
		def ret_fn2(slow_ser, fast_ser):
			return ret_fn(slow_ser, fast_ser)
	else:
		def ret_fn2(slow_ser, fast_ser):
			ret = ret_fn(slow_ser, fast_ser)
			return ser_range_center_clip(ret, thresh, inner=SIDEWAYS, outer=clip, inclusive=False)

	return ret_fn2

def expanding_returnify(ret_type, thresh=None, clip=False, agg_freq='cal_daily'):
	"""
	Expanding spread, regular return, or log return.
	"""
	ret_fn = returnify(ret_type, thresh=thresh, clip=clip)
	agg_freq = RUNT_FREQ_TRANSLATOR[agg_freq]

	def retx(slow_ser, fast_ser):
		first_slow = slow_ser.dropna().groupby(pd.Grouper(freq=agg_freq)).transform(single_row_filter('f'))
		return ret_fn(first_slow, fast_ser.dropna())

	return retx

def variable_expanding_returnify(ret_type, stat_type, clip=False, thresh_scalar=1, agg_freq='cal_daily'):
	"""
	Expanding return thresholded on past period statistic.
	"""
	ret_fn = expanding_returnify(ret_type, thresh=None, clip=False, agg_freq=agg_freq)
	stat_fn = statistic(stat_type, abs_val=True, agg_freq=agg_freq)
	agg_freq = RUNT_FREQ_TRANSLATOR[agg_freq]

	def vretx(slow_ser, fast_ser):
		ret = ret_fn(slow_ser, fast_ser)
		stat = stat_fn(ret) * thresh_scalar
		thresh = pd_slot_shift(pd.DataFrame({0:-stat, 1: stat}, index=ret.index), periods=1, freq=agg_freq)
		idx = ret.index & thresh.index
		clipped = ser_range_center_clip(ret.loc[idx], thresh.loc[idx], inner=SIDEWAYS, outer=clip, inclusive=False)
		return clipped

	return vretx

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
	return lambda ser: ser.dropna().rolling(window=len(weights), min_periods=len(weights)).apply(dot_weights_fn)

NORM_FN_MAP = {
	'dzn': lambda ser: (ser-ser.mean()) / ser.std(),
	'dmx': lambda ser: 2 * ((ser-ser.min()) / (ser.max()-ser.min())) - 1
}

def normalize(norm_type):
	return lambda ser: ser.transform(NORM_FN_MAP[norm_type])

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

	logging.debug('breakpoints: {}'.format(str(breakpoints)))
	logging.debug('symbols: {}'.format(str(symbols)))

	return lambda ser: ser.transform(encoder)


""" ********** FILTERS ********** """
DEFAULT_RET_IDX = False 	# Default to return value instead of index
DEFAULT_IDX_NORM = False	# Default to return index value instead of normalized score
DEFAULT_FNZ_IDX_SHF = 1		# Default to return index shifted by +1

def first_only(ser, ret_idx=DEFAULT_RET_IDX):
	idx = ser.first_valid_index()

	if (idx is None or ret_idx):
		return idx
	else:
		return ser.loc[idx]

def first_nonzero(ser, ret_idx=DEFAULT_RET_IDX, idx_norm=DEFAULT_IDX_NORM):
	idx = arr_nonzero(ser.values, ret_idx=ret_idx, idx_norm=idx_norm, idx_shf=DEFAULT_FNZ_IDX_SHF)

	if (not isinstance(idx, np.ndarray) and (idx is None or idx == 0)):
		return idx
	else:
		return idx.item(0)

def last_only(ser, ret_idx=DEFAULT_RET_IDX):
	idx = ser.last_valid_index()

	if (idx is None or ret_idx):
		return idx
	else:
		return ser.loc[idx]

def nth_only(ser, ret_idx=DEFAULT_RET_IDX):
	if (ret_idx):
		return ser.nth(specifier).index
	else:
		return ser.nth(specifier)

SRF_SPECIFIER = {
	'f': first_only,
	'fnz': first_nonzero,
	'fnzi_normed': partial(first_nonzero, ret_idx=True, idx_norm=True),
	'l': last_only,
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

	return compose(flt, mp)


""" ********** JSON-STR-TO-CODE TRANSLATORS ********** """
RUNT_FN_TRANSLATOR = {
	"diff": difference,
	"ret": returnify,
	"retx": expanding_returnify,
	"vretx": variable_expanding_returnify,
	"fracdiff": fracdiff,
	"smw": simple_moving_window,
	"xw": expanding_window,
	"emw": exponential_moving_window,
	"norm": normalize,
	"sym": symbolize,
	"srf": single_row_filter,
	"sgn": signify
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
	DT_HOURLY_FREQ: DT_HOURLY_FREQ,
	DT_CAL_DAILY_FREQ: DT_CAL_DAILY_FREQ,
	DT_BIZ_DAILY_FREQ: DT_BIZ_DAILY_FREQ,
	None: None
}
