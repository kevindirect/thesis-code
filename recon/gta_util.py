"""
Kevin Patel
"""

import sys
import os
from os import sep
from os.path import splitext
import logging

import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, inner_join, zdiv, benchmark
from recon.common import REPORT_DIR
from recon.label_util import shift_label


""" ********** APPLY FUNCTIONS ********** """
def feat_label_apply(feat_df, feat_df_desc, label_df, apply_fn, label_shift=-1):
	"""
	Return matrix of test result for a feature/label apply function.
	
	Args:
		feat_df (pd.DataFrame): df of feature series
		feat_df_desc (str): string that uniquely identifies feat_df
		label_df (pd.DataFrame): df of label series
		apply_fn (function): function to apply, the only not set parameters must be two series

	"""
	result = pd.DataFrame()

	for feat in feat_df.columns:
		row = {}
		row['feat_col_name'] = feat
		row.update({label: apply_fn(feat_df[feat], shift_label(label_df[label], shift_periods=label_shift)) for label in label_df.columns})
		result = result.append(row, ignore_index=True)

	if (feat_df_desc is not None):
		result.insert(0, 'feat_df_desc', feat_df_desc)

	return result


""" ********** TEST FUNCTIONS ********** """
def corr(a, b, method='pearson'):
	"""
	Return correlation of series a and b.
	Method can be 'pearson', 'spearman', or 'kendall'.
	"""
	return a.corr(b, method=method)

def count(a, b, method='ratio'):
	"""
	Return count of rows of a relative to b (or non-null count of a if count method is selected).

	Examples:
	a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='a')										# 10 non-null
	b = pd.Series([0, 1, None,	3, 4, None, 6, 7, 8, 9], name='b')								# 8 non-null
	c = pd.Series([None, None, None, None, None, None, None, None, None, None], name='c')		# 0 non-null
	d = pd.Series([0, 1, 2, 3,	4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], name='d')				# 15 non-null
	e = pd.Series([0, 1, 2, None, 4, None, 6, 7, 8, 9, 10, 11, 12, None, 14], name='e')			# 12 non-null
	
	count(a, b) # expected 1
	count(b, a) # expected .8

	count(a, c) # expected 0
	count(c, a) # expected 0

	count(a, d) # expected .667
	count(d, a) # expected 1

	count(a, e) # expected .667
	count(e, a) # expected .8

	count(b, d) # expected .533
	count(d, b) # expected 1

	count(b, e) # expected .583
	count(e, b) # expected .875

	count(d, e) # expected 1
	count(e, d) # expected .8
	"""
	if (method == 'ratio'):
		adf, bdf = a.dropna().to_frame(), b.dropna().to_frame()
		common_count, target_count = inner_join(adf, bdf).index.size, bdf.index.size
		return zdiv(common_count, target_count)
	elif (method == 'count'):
		return a.count()


""" ********** MISC UTIL ********** """
report_path_dir = lambda dataset_fname, asset: sep.join([REPORT_DIR, splitext(dataset_fname)[0], asset]) +sep


""" ********** JSON-STR-TO-CODE TRANSLATORS ********** """
GTA_TYPE_TRANSLATOR = {
	"fl": feat_label_apply
}

GTA_TEST_TRANSLATOR = {
	"corr": corr,
	"count": count
}
