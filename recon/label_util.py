# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import DT_CAL_DAILY_FREQ, remove_dups_list, list_get_dict, get_subset
from mutate.label import LABEL_SFX_LEN
from recon.common import dum

UP, DOWN, SIDEWAYS = 1, -1, 0


""" ********** FORECAST MASKS ********** """
def eod_fct(label_df, col=0, eod_thresh=(0, 0)):
	"""
	EOD (End of Day) Forecast:
		Return the simple end of day forecast of an expanding return
	"""
	if (isinstance(col, int)):
		col_name = label_df.columns[0]
	elif (isinstance(col, str)):
		col_name = col
	else:
		col_name = col

	lbl = label_df[col_name].copy()
	lbl[lbl > eod_thresh[1]] = UP
	lbl[lbl < eod_thresh[0]] = DOWN
	lbl[~pd.isnull(lbl) & ~lbl.isin((UP, DOWN))] = SIDEWAYS

	return lbl.dropna().to_frame()

def default_fct(label_df, name_pfx=''):
	"""
	Default Forecast (Fast Break + EOD):
		The label is the sign of the first thresh break, otherwise the label is the EOD sign change
	"""
	return label_df


def break_fct(label_df, name_pfx='', lower_bound=None, upper_bound=None):
	"""
	Break Forecast:
		The label is the sign of the first thresh break if it is between the upper and lower bounds,
		otherwise the label is 0
	"""
	lbl = label_df.copy()
	dir_name, brk_name = '_'.join([name_pfx, 'dir']), '_'.join([name_pfx, 'brk'])

	if (lower_bound is not None and isinstance(lower_bound, int)):
		lbl.loc[lbl[brk_name] < lower_bound, dir_name] = SIDEWAYS

	if (upper_bound is not None and isinstance(upper_bound, int)):
		lbl.loc[lbl[brk_name] > upper_bound, dir_name] = SIDEWAYS

	return lbl


def fastbreak_fct(label_df, name_pfx='', velocity=False):
	"""
	Fast Break Forecast:
		The label is the direction or velocity of the first thresh break, otherwise the label is 0
	"""
	filtered = break_fct(label_df, name_pfx=name_pfx, lower_bound=1)

	if (velocity):
		dir_name, brk_name = '_'.join([name_pfx, 'dir']), '_'.join([name_pfx, 'brk'])

		# Map break times to integer min rank in inverse order to get mangitudes, multiply by direction for velocity
		speed = filtered.loc[:, brk_name].rank(axis=0, method='dense', na_option='keep', ascending=False, pct=False)
		filtered.loc[:, dir_name] = filtered.loc[:, dir_name] * speed

	return filtered


def confidence_fct(label_df, name_pfx='', magnitude=False):
	"""
	Confidence Weight Forecast:
		Set the label to the sign of the break whose frequency exceeds conf_thresh

	Args:
		magnitude (bool): scale by number of breaks in labelled direction
	"""
	lbl = label_df.copy()
	dir_name, brk_name = '_'.join([name_pfx, 'dir']), '_'.join([name_pfx, 'brk'])
	nmb_name, nmt_name = '_'.join([name_pfx, 'nmb']), '_'.join([name_pfx, 'nmt'])

	# No breaks, set to sideways (sets 0 thresh eod to zero)
	lbl.loc[lbl[nmt_name] == 0, dir_name] = SIDEWAYS

	# Most common break dir is opposite to first break dir, reverse dir. The (lbl[nmb_name]/lbl[nmt_name]) >= .5 case is already set correctly
	lbl.loc[(lbl[nmt_name] > 0) & ((lbl[nmb_name]/lbl[nmt_name]) < .5), dir_name] = -lbl.loc[:, dir_name]

	if (magnitude):
		total_breaks = lbl.loc[lbl[nmt_name] > 0, nmt_name]
		breaks_with_fb = lbl.loc[lbl[nmt_name] > 0, nmb_name]
		confidence = np.maximum(breaks_with_fb, total_breaks-breaks_with_fb)
		lbl.loc[lbl[nmt_name] > 0, dir_name] = lbl.loc[:, dir_name] * confidence

	return lbl


def fastbreak_confidence_fct(label_df, name_pfx='', momentum=False):
	"""
	Fast Break Confidence Forecast:
		The label is the sign of the first thresh break only if it is the most common break,
		otherwise the label is 0
	"""
	dir_name = '_'.join([name_pfx, 'dir'])
	fb = fastbreak_fct(label_df, name_pfx=name_pfx, velocity=momentum)
	cf = confidence_fct(label_df, name_pfx=name_pfx, magnitude=momentum)

	if (momentum):
		# Velocity (Fastbreak) and Mass (Confidence) in agreement: P = M * V
		fb.loc[np.sign(fb[dir_name]) == np.sign(cf[dir_name]), dir_name] = fb.loc[:, dir_name] * cf.loc[:, dir_name].abs()

	# Fastbreak and Confidence not in agreement, set label to sideways
	fb.loc[np.sign(fb[dir_name]) != np.sign(cf[dir_name]), dir_name] = SIDEWAYS

	return fb


""" ********** LABELUTIL FUNCTIONS ********** """
def shift_label(label_ser, shift_periods=-1):
	return label_ser.dropna().shift(periods=shift_periods, freq=None, axis=0).dropna().astype(int)

def get_base_labels(df_columns):
	return remove_dups_list([col_name[:-LABEL_SFX_LEN] for col_name in df_columns])

make_sw_dict = lambda sw: {"exact": [], "startswith": [sw], "endswith": [], "regex": [], "exclude": None}

def gen_label_dfs(lab_dict, lab_paths, asset_name, forecast_mask=default_fct):

	for lab_path in filter(lambda lab_path: lab_path[0]==asset_name, lab_paths):
		lab_df = list_get_dict(lab_dict, lab_path)
		yield apply_label_mask(lab_df, forecast_mask=forecast_mask)

def apply_label_mask(lab_df, forecast_mask, normalize_idx=True):
	lab_fct_df = pd.DataFrame(index=lab_df.index)

	logging.debug('base_labels: ' +', '.join(get_base_labels(lab_df.columns[1:])))
	logging.debug('all cols: ' +', '.join(lab_df.columns))

	label_col_sel = {base_label: get_subset(lab_df.columns[1:], make_sw_dict(base_label))
		for base_label in get_base_labels(lab_df.columns[1:])}
	logging.debug(label_col_sel)

	# Iterate through all thresholded variations of this label
	for base_label, base_label_cols in label_col_sel.items():
		logging.debug('base label: ' +base_label)
		dir_col_name = '_'.join([base_label, 'dir'])
		fct_df = forecast_mask(lab_df[base_label_cols].dropna(axis=0, how='all'), name_pfx=base_label)
		lab_fct_df[dir_col_name] = fct_df[dir_col_name].dropna()

	if (normalize_idx):
		lab_fct_df.index = lab_fct_df.index.normalize()

	return lab_fct_df
