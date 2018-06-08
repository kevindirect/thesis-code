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


def default_fct(label_df, name_pfx=''):
	"""
	Default Forecast (Fast Break + EOD):
		The label is the sign of the first thresh break,
		otherwise the label is the EOD sign change
	"""
	return label_df


def break_fct(label_df, name_pfx='', lower_bound=None, upper_bound=None):
	"""
	Break Forecast:
		The label is the sign of the first thresh break if it is between the upper and lower bounds,
		otherwise the label is 0
	"""
	dir_name, brk_name = '_'.join([name_pfx, 'dir']), '_'.join([name_pfx, 'brk'])

	if (lower_bound is not None and isinstance(lower_bound, int)):
		label_df[label_df[brk_name] < lower_bound][dir_name] = 0

	if (upper_bound is not None and isinstance(upper_bound, int)):
		label_df[label_df[brk_name] > upper_bound][dir_name] = 0

	return label_df


def fastbreak_fct(label_df, name_pfx='', upper_bound=None):
	"""
	Fast Break Forecast:
		The label is the sign of the first thresh break,
		otherwise the label is 0
	"""
	return break_fct(label_df, name_pfx=name_pfx, lower_bound=1, upper_bound=upper_bound)


def confidence_fct(label_df, name_pfx=''):
	"""
	Confidence Weight Forecast:
		The label is the sign of the most common break,
		if there is none (either equally many or no breaks) the label is 0
	"""
	dir_name, brk_name = '_'.join([name_pfx, 'dir']), '_'.join([name_pfx, 'brk'])
	nmb_name, nmt_name = '_'.join([name_pfx, 'nmb']), '_'.join([name_pfx, 'nmt'])

	# Number with and against the direction of first break
	number_with_fb = label_df[label_df[brk_name] > 0][nmb_name]
	number_against_fb = label_df[label_df[brk_name] > 0][nmt_name] - number_toward_fb

	# Set locations where there are no breaks, or equal number both ways to 0
	label_df[label_df['nmt'] == 0][dir_name] = 0
	label_df[number_opposite_fb == number_toward_fb][dir_name] = 0

	# Reverse the prediction wherever the first break direction is not the most populous
	# (the case where the first break is the most populous is already correctly labelled)
	label_df[number_opposite_fb > number_toward_fb][dir_name] = -label_df[dir_name]

	return label_df


def fastbreak_confidence_fct(label_df, name_pfx='', upper_bound=None):
	"""
	Fast Break Confidence Forecast:
		The label is the sign of the first thresh break if it is the most common break,
		otherwise the label is 0
	"""
	dir_name = '_'.join([name_pfx, 'dir'])
	fb = fastbreak_fct(label_df, name_pfx=name_pfx, upper_bound=upper_bound)
	cw = confidence_fct(label_df, name_pfx=name_pfx)

	# Set locations in fast break where it is not the most populous to 0
	fb[fb[dir_name] != cw[dir_name]][dir_name] = 0

	return fb
