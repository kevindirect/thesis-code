"""
Kevin Patel
"""
import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, get_custom_biz_freq
from mutate.common import STANDARD_DAY_LEN


gaussian_breakpoints = {
	2 : [0],
	3 : [-0.43, 0.43],
	4 : [-0.67, 0, 0.67],
	5 : [-0.84, -0.25, 0.25, 0.84],
	6 : [-0.97, -0.43, 0, 0.43, 0.97],
	7 : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
	8 : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
	9 : [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
	10: [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
	11: [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
	12: [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
	13: [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
	14: [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
	15: [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
	16: [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
	17: [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56],
	18: [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59],
	19: [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62],
	20: [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]
}

MIN_VAL_OUT, MAX_VAL_OUT = -1, 1	# Inclusive limits of range for min-max scaling
uniform_breakpoints = {n: [MIN_VAL_OUT+(k*((MAX_VAL_OUT-MIN_VAL_OUT)/n)) for k in range(1, n)] for n in range(2, 21)}

BREAKPOINT_MAPPING = {
	"gaussian": gaussian_breakpoints,
	"uniform": uniform_breakpoints
}


""" ********** MISC ********** """
def most_freq_subseq_len(df, capture=.95):
	cust, count_df = cust_count(df)
	vc = count_df.apply(pd.Series.value_counts, normalize=True)
	proportion = 0.0

	for idx, val in sorted(list(vc.iteritems()), key=lambda tup: tup[0], reverse=True):
		print(idx)
		if (proportion > capture):
			return {'max':ser.idxmax(), proportion: idx}
		else:
			proportion += val


""" ********** NORMALIZATION ********** """
zscore_transform = lambda ser: (ser-ser.mean()) / ser.std()
bipolar_mm_transform = lambda ser: 2 * ((ser-ser.min()) / (ser.max()-ser.min())) - 1

NORM_FUN_MAPPING = {
	'dzn': lambda ser: (ser-ser.mean()) / ser.std(),
	'dmx': lambda ser: 2 * ((ser-ser.min()) / (ser.max()-ser.min())) - 1
}

def day_norm(df, transform_fn, freq=DT_CAL_DAILY_FREQ):
	if (freq is None):
		freq = get_custom_biz_freq(df)
	return df.groupby(pd.Grouper(freq=freq)).transform(transform_fn)


""" ********** SEGMENTATION ********** """
DEF_PATTERN_SIZE = 5
DEF_PIP_METHOD = 'vd'

def pip_df(df, pattern_size=DEF_PATTERN_SIZE, method=DEF_PIP_METHOD):
	"""
	Perceptually Important Points: Choose points by distance to points already in pattern.

	This implementation assumes all points are equally spaced.
	"""
	distance_fun = {
		'ed': None,	# euclidean distance
		'pd': None,	# perpendicular distance to joining line
		'vd': vd	# vertical distance to joining line
	}.get(method)

	def get_max_dist_point(ser, left, right, dist_fun=distance_fun):
		"""
		Loop through all points in ser from left to right ilocs,
		get max point and return it's iloc.
		"""
		if (left == right):
			raise ValueError('Zero Division')

		md, md_iloc = 0, 0
		left_xy = (left, ser.iloc[left])
		right_xy = (right, ser.iloc[right])

		for point in ser.iloc[left:right].iteritems():
			point_xy = (ser.index.get_loc(point[0]), point[1])
			dist = dist_fun(left_xy, right_xy, point_xy)
			if (md < dist):
				md = dist
				md_iloc = point_xy[0]

		return md, md_iloc

	def get_next_point(ser, pattern):
		md, md_iloc = 0, 0

		for l, r in pairwise(pattern):
			dist, iloc = get_max_dist_point(ser, l, r)
			if (md < dist):
				md = dist
				md_iloc = iloc

		return md_iloc

	def pip_ser(ser, pattern_size=pattern_size):
		if (len(ser) > pattern_size):
			pattern_iloc = SortedSet([0, len(ser)-1])
			while (len(pattern_iloc) < pattern_size):
				next_point = get_next_point(ser, pattern_iloc)
				pattern_iloc.add(next_point)
			pattern = ser.iloc[pattern_iloc]

		elif (len(ser) == pattern_size):
			pattern = ser
		else:
			pattern = ser
		return pattern

	# TODO - HANDLE ALL DAYS WITH LESS THAN pattern_size number of points
	# Options:
	#     - greater: run pip
	#     - equal: no change or drop
	#     - less than: drop
	# Set distance function to use

	cust = get_custom_biz_freq(df)
	return df.groupby(pd.Grouper(freq=cust)).transform(pip_ser)



""" ********** SYMBOLIZATION ********** """
NULL_VALUE_SYMBOL = '_'

def get_sym_list(breakpoints, numeric_symbols=True):
	"""
	Return list of symbols based on provided breakpoints list.
	"""
	if (numeric_symbols):
		return [str(idx+1) for idx in range(len(breakpoints)+1)]
	else:
		return list(map(lambda idx: chr(ord('a') +idx), range(len(breakpoints)+1)))

def symbolize_value(value, breakpoints, symbols):
	"""
	Return value converted to symbol based on provided breakpoints and symbols.
	"""
	for idx, brk in enumerate(breakpoints):
		if (value <= brk):
			return symbols[idx]
	else:
		if (value > breakpoints[-1]):
			return symbols[-1]
		else:
			return NULL_VALUE_SYMBOL

def clamp_subseq_len(subseq, max_seg=STANDARD_DAY_LEN):
	"""
	Clamp subsequence length to max_seg value.

	Args:
		subseq (pd.Series): subsequence of values
		max_seg (int): max number of letters per word

	Return:
		clamped length pd.Series
	"""
	if (max_seg is None):
		return subseq
	else:
		subseq_len = subseq.shape[0]

		if (subseq_len > max_seg):
			if (max_seg == STANDARD_DAY_LEN): 	# Assumes any extra data is pre-market trading
				segs = subseq.tail(max_seg)

			elif (max_seg < STANDARD_DAY_LEN):
				if (subseq_len == STANDARD_DAY_LEN):
					# XXX - information is lost in this case
					segs = subseq.tail(max_seg)
				elif (subseq_len > STANDARD_DAY_LEN):
					segs = subseq.tail(max_seg)

		return segs

def encode_subseq(subseq, max_seg, breakpoints, symbols):
	"""
	Encode float series -> symbol series.
	Does not perform paa or any other subseries downsampling/aggregation.

	Args:
		subseq (pd.Series):
		max_seg (int): max number of letters per word
		breakpoints (list):
		symbols (list):

	Return:
		pd.Series with symbols instead of values
	"""
	clamped = clamp_subseq_len(subseq, max_seg=max_seg)
	sym_map = partial(symbolize_value, breakpoints=breakpoints, symbols=symbols)
	code = clamped.map(sym_map).str.cat(sep=',')

	return code

def encode_df(df, breakpoint_dict, num_sym, numeric_symbols=True):
	"""
	Encode float df -> symbol df.
	Does not perform paa or any other subseries downsampling/aggregation.

	Args:
		df (pd.DataFrame):
		breakpoint_dict (dict):
		num_sym (int): alphabet size
		max_seg (int): max number of letters per word
		numeric_symbols (boolean): numeric or non-numeric symbols

	Return:
		pd.Series with symbols instead of values
	"""
	breakpoints = breakpoint_dict[num_sym]
	symbols = get_sym_list(breakpoints, numeric_symbols=numeric_symbols)
	logging.debug('breakpoints: ' +str(breakpoints))
	logging.debug('symbols: ' +str(symbols))

	encoder = partial(symbolize_value, breakpoints=breakpoints, symbols=symbols)
	mapped = df.applymap(encoder)

	return mapped


# def sax_df(df, num_sym, max_seg=None, numeric_symbols=True):
# 	"""
# 	Symbolic Aggregate Approximation (SAX) style symbolization.
# 	This does not perform paa or any other subseries downsampling/aggregation
# 	on the data.

# 	Args:
# 		df (pd.Dataframe):
# 		num_sym (int): alphabet size

# 	Return:
# 		pd.Dataframe with rows aggregated by day into symbolic sequences
# 	"""
# 	gaussian_brks = gaussian_breakpoints[num_sym]
# 	gaussian_syms = get_sym_list(gaussian_brks, numeric_symbol=numeric_symbol)

# 	def sax_ser(ser):
# 		day_len = ser.shape[0]

# 		if (max_seg is not None and max_seg < day_len):
# 			if (max_seg == STANDARD_DAY_LEN):
# 				# Assumes any day beyond the standard length is due to pre-market trading
# 				segs = ser.tail(max_seg)

# 			elif (max_seg < STANDARD_DAY_LEN):
# 				if (day_len == STANDARD_DAY_LEN):
# 					# XXX - information is lost in this case
# 					segs = ser.tail(max_seg)
# 				elif (day_len > STANDARD_DAY_LEN):
# 					segs = ser.tail(max_seg)
# 		else:
# 			segs = ser
# 		code = segs.map(symbolize_value, gaussian_brks, gaussian_syms).str.cat(sep=',')
# 		return code

# 	cust = get_custom_biz_freq(df)
# 	# XXX - Known Issue: some thresh group data is lost after saxify (rows that are not non-null in all columns)
# 	# UPDATE: This is probably an issue with normalize, not with saxing
# 	saxed = df.groupby(pd.Grouper(freq=cust)).aggregate(sax_ser)

# 	return saxed


""" ********** CLUSTERING ********** """
"""
K Means Clustering

Requirements:
	- Stationary Intraday Time Series

"""

"""
PCA of all hours, take top 1-3 components

Gives us the linear combination of hours that account for most of the variance
"""

"""
Take the slope of a best fit line through the id time series

"""



