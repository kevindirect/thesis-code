"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from dask import delayed

from common_util import DT_CAL_DAILY_FREQ, remove_dups_list, list_get_dict, get_subset
from mutate.common import dum
from mutate.label import LABEL_SFX_LEN

# All functions in this module are deprecated

""" ********** DIRECTION SYMBOLS ********** """
UP, DOWN, SIDEWAYS = 1, -1, 0


""" ********** FORECAST MASKS ********** """
def fastbreak_eod_fct(label_df, name_pfx=''):
	"""
	Default Forecast (Fast Break + EOD):
		The label is the sign of the first thresh break, otherwise the label is the EOD sign change
	"""
	return label_df


def fastbreak_fct(ser, thresh):
	"""
	Default Forecast (Fast Break + EOD):
		The label is the sign of the first thresh break, otherwise the label is the EOD sign change
	"""
	ser[ser > thresh]


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


""" ********** FORECAST APPLY ********** """
def extract_mask_field(lab_df, mask_field='mag', normalize_idx=True):
	extracted_df = pd.DataFrame(index=lab_df.index)

	logging.debug('base_labels: '.format(', '.join(get_base_labels(lab_df.columns[1:]))))
	logging.debug('all cols: {}'.format(', '.join(lab_df.columns)))

	label_col_sel = {base_label: get_subset(lab_df.columns[1:], make_sw_search_dict(base_label))
		for base_label in get_base_labels(lab_df.columns[1:])}
	logging.debug(label_col_sel)

	# Iterate through all thresholded variations of this base label
	for base_label, base_label_cols in label_col_sel.items():
		logging.debug('base label: {}'.format(base_label))
		dir_col_name = '_'.join([base_label, mask_field])
		extracted_df = lab_df[dir_col_name].dropna(axis=0, how='all')

	if (normalize_idx):
		lab_fct_df.index = lab_fct_df.index.normalize()

	return lab_fct_df

def apply_label_mask(lab_df, forecast_mask, normalize_idx=True):
	lab_fct_df = pd.DataFrame(index=lab_df.index)

	logging.debug('base_labels: '.format(', '.join(get_base_labels(lab_df.columns[1:]))))
	logging.debug('all cols: {}'.format(', '.join(lab_df.columns)))

	label_col_sel = {base_label: get_subset(lab_df.columns[1:], make_sw_search_dict(base_label))
		for base_label in get_base_labels(lab_df.columns[1:])}
	logging.debug(label_col_sel)

	# Iterate through all thresholded variations of this base label
	for base_label, base_label_cols in label_col_sel.items():
		logging.debug('base label: {}'.format(base_label))
		dir_col_name = '_'.join([base_label, 'dir'])
		fct_df = forecast_mask(lab_df[base_label_cols].dropna(axis=0, how='all'), name_pfx=base_label)
		lab_fct_df[dir_col_name] = fct_df[dir_col_name].dropna()

	if (normalize_idx):
		lab_fct_df.index = lab_fct_df.index.normalize()

	return lab_fct_df

make_sw_search_dict = lambda sw: {"exact": [], "startswith": [sw], "endswith": [], "regex": [], "exclude": None}

def get_base_labels(df_columns):
	return remove_dups_list([col_name[:-LABEL_SFX_LEN] for col_name in df_columns])


# """ ********** LABEL EXTRACTION / PREPARATION FUNCTIONS ********** """
def prep_labels(label_df, types=['bool', 'int']):
	"""
	Take label df and apply masks to produce df of label series.
	"""
	gb_label_df = delayed(lambda d: d.groupby(pd.Grouper(freq=DT_CAL_DAILY_FREQ)).last())(label_df)
	label_groups = []

	if ('bool' in types):
		eod0 = delayed(eod_fct)(gb_label_df, eod_thresh=0).add_suffix('_eod(0%)')
		eod1 = delayed(eod_fct)(gb_label_df, eod_thresh=.01).add_suffix('_eod(1%)')
		eod2 = delayed(eod_fct)(gb_label_df, eod_thresh=.02).add_suffix('_eod(2%)')
		fbeod = delayed(apply_label_mask)(gb_label_df, fastbreak_eod_fct).add_suffix('_fbeod')
		fb = delayed(apply_label_mask)(gb_label_df, fastbreak_fct).add_suffix('_fb')
		conf = delayed(apply_label_mask)(gb_label_df, confidence_fct).add_suffix('_conf')
		fbconf = delayed(apply_label_mask)(gb_label_df, fastbreak_confidence_fct).add_suffix('_fbconf')
		label_groups.extend((eod0, eod1, eod2, fbeod, fb, conf, fbconf))

	if ('int' in types):
		vel = delayed(apply_label_mask)(gb_label_df, partial(fastbreak_fct, velocity=True)).add_suffix('_vel')
		mag = delayed(apply_label_mask)(gb_label_df, partial(confidence_fct, magnitude=True)).add_suffix('_mag')
		mom = delayed(apply_label_mask)(gb_label_df, partial(fastbreak_confidence_fct, momentum=True)).add_suffix('_mom')
		label_groups.extend((vel, mag, mom))

	labels = delayed(reduce)(outer_join, label_groups)
	labels = delayed(df_dti_index_to_date)(labels)

	return labels
