# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from numba import jit, vectorize
from dask import delayed, compute

from common_util import DT_CAL_DAILY_FREQ, search_df, chained_filter
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.ops import *


# def par_gb_apply(groupby_obj, func):
# 	retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in groupby_obj)
# 	return pd.concat(retLst)

def day_change(ser, len=1):
	return ser.first() - ser.last()

def diff_change(ser):
	return pd.Series(np.sign(ser.diff())).to_string()

@delayed
def parsac(df, fn, gb=pd.Grouper(freq=DT_CAL_DAILY_FREQ), rc=False, **kwargs):
	"""
	Dask parallelized and lazy evaluated split-apply-combine procedure over all series in df.

	Args:
		df (pd.DataFrame): DataFrame to sac
		fn (function): function to apply to all series or groupby-ed series
		gb (Optional): defines the groupby operation. If None, no groupby is performed
		rc (boolean): return computed result or not
		kwargs: keyword args to fn

	Returns:
		pd.Dataframe
	"""
	result = {}
	col_names = df.columns

	# Split (optional)
	if (gb is not None):
		df = delayed(df.groupby)(gb)

	# Apply to all series or groupby-ed series
	for col_name in col_names:
		result[col_name] = delayed(fn)(df[col_name], **kwargs)

	# Combine (unpack)
	if (rc):
		result = pd.DataFrame.from_dict(compute(result)[0])
	else:
		result = delayed(compute)(result)
		result = delayed(pd.DataFrame.from_dict)(result[0])
	return result


def sac(df, fn, gb=pd.Grouper(freq=DT_CAL_DAILY_FREQ), **kwargs):
	"""
	Split-apply-combine procedure over all series in df.

	Args:
		df (pd.DataFrame): DataFrame to sac
		fn (function): function to apply to all series or groupby-ed series
		gb (Optional): defines the groupby operation. If None, no groupby is performed
		kwargs: keyword args to fn

	Returns:
		pd.Dataframe
	"""
	result = {}
	col_names = df.columns

	# Split (optional)
	if (gb is not None):
		df = df.groupby(gb)

	# Apply to all series or groupby-ed series
	for col_name in col_names:
		result[col_name] = fn(df[col_name], **kwargs)

	# Combine (unpack)
	return pd.DataFrame.from_dict(result)

