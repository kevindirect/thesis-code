"""
Kevin Patel
"""

import sys
import os
from os import sep
from os.path import splitext
import logging

import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, inner_join, right_join, benchmark
from recon.common import REPORT_DIR
from recon.label_util import shift_label


""" ********** APPLY FUNCTIONS ********** """
def feat_label_apply(feat_df, label_df, apply_fn, label_shift=-1):
	"""
	Return matrix of test result for a feature/label apply function.
	
	Args:
		feat_df (pd.DataFrame): features
		label_df (pd.DataFrame): labels
		apply_fn (function): function to apply, the only not set parameters must be two series

	"""
	result = pd.DataFrame()

	for feat in feat_df.columns:
		row = {}
		row['feat_col_name'] = feat
		row.update({label: apply_fn(feat_df[feat], shift_label(label_df[label], shift_periods=label_shift)) for label in label_df.columns})
		result = result.append(row, ignore_index=True)

	return result


""" ********** TEST FUNCTIONS ********** """
def corr(a, b, method='pearson'):
	"""
	Return correlation of series a and b.
	Method can be 'pearson', 'spearman', or 'kendall'.
	"""
	return a.corr(b, method=method)

def count(a, b, method='ratio'): # TODO
	"""
	Return count of rows of a relative to b.
	Type can be 'ratio' or 'count'.
	"""
	if (method == 'ratio'):
		return float(inner_join(a.dropna(), b.dropna()).index.size / b.dropna().index.size)
	# a = a.to_frame()
	# b = b.to_frame()

	# if (method == 'ratio'):
	# 	((a.isnull() && b.isnull()) == True).count()
	# elif (method == 'count'):
	# 	return inner_join(a, b).count()


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
