# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, DT_CAL_DAILY_FREQ, search_df, get_custom_biz_freq, dti_to_ymd, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum


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
def sax_df(df, num_sym, max_seg=None, numeric_symbols=True):
	"""
	Symbolic Aggregate Approximation (SAX) style symbolization.
	This does not perform paa or any other subseries downsampling/aggregation
	on the data.

	Args:
		df (pd.Dataframe):
		num_sym (int): alphabet size

	Return:
		pd.Dataframe with rows aggregated by day into symbolic sequences
	"""
	breakpoints = gaussian_breakpoints[num_sym]
	if (numeric_symbols):
		symbols = [str(idx+1) for idx in range(len(breakpoints)+1)]
	else:
		symbols = list(map(lambda idx: chr(ord('a') +idx), range(len(breakpoints)+1)))

	def symbolize_value(value):
		for idx, brk in enumerate(breakpoints):
			if (value <= brk):
				return symbols[idx]
		else:
			return symbols[len(breakpoints)]

	def sax_ser(ser):
		if (max_seg is not None and ser.shape[0] > max_seg):
			segs = ser.tail(max_seg)
		else:
			segs = ser
		code = segs.map(symbolize_value).str.cat(sep=',')
		return code

	cust = get_custom_biz_freq(df)
	saxed = df.groupby(pd.Grouper(freq=cust)).aggregate(sax_ser)

	return saxed




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




















