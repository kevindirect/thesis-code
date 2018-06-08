# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_util import remove_dups_list
from mutate.label import LABEL_SFX_LEN
from recon.common import dum


def get_base_labels(df_columns):
	return remove_dups_list([col_name[:-LABEL_SFX_LEN] for col_name in df_columns])


"""
Default Prediction Objective (Fast Break + EOD):
	The label is the sign of the first thresh break,
	otherwise the label is the EOD sign change
"""


def break_po(label_df, lower_bound=None, upper_bound=None):
	"""
	Break Prediction Objective:
		The label is the sign of the first thresh break if it is between the upper and lower bounds,
		otherwise the label is 0
	"""
	if (lower_bound is not None and isinstance(lower_bound, int)):
		label_df[label_df['brk'] < lower_bound]['dir'] = 0

	if (upper_bound is not None and isinstance(upper_bound, int)):
		label_df[label_df['brk'] > upper_bound]['dir'] = 0

	return label_df


def fast_break_po(label_df, upper_bound=None):
	"""
	Fast Break Prediction Objective:
		The label is the sign of the first thresh break,
		otherwise the label is 0
	"""
	return break_po(label_df, lower_bound=1, upper_bound=upper_bound)


def confidence_weight_po(label_df):
	"""
	Confidence Weight Prediction Objective:
		The label is the sign of the most common break,
		if there is none (either equally many or no breaks) the label is 0
	"""
	# Number with and against the direction of first break
	number_with_fb = label_df[label_df['brk'] > 0]['nmb']
	number_against_fb = label_df[label_df['brk'] > 0]['nmt'] - number_toward_fb

	# Set locations where there are no breaks, or equal number both ways to 0
	label_df[label_df['nmt'] == 0]['dir'] = 0
	label_df[number_opposite_fb == number_toward_fb]['dir'] = 0

	# Reverse the prediction wherever the first break direction is not the most populous
	# (the case where the first break is the most populous is already correctly labelled)
	label_df[number_opposite_fb > number_toward_fb]['dir'] = -label_df['dir']

	return label_df


def confidence_fast_break_po(label_df, upper_bound=None):
	"""
	Confidence Fast Break Prediction Objective:
		The label is the sign of the first thresh break if it is the most common break,
		otherwise the label is 0
	"""
	fb = fast_break_po(label_df, upper_bound=upper_bound)
	cw = confidence_weight_po(label_df)

	# Set locations in fast break where it is not the most populous to 0
	fb[fb['dir'] != cw['dir']]['dir'] = 0

	return fb
