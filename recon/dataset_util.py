"""
Kevin Patel
"""

import sys
import os
import logging
from functools import partial, reduce

import pandas as pd
from dask import delayed

from common_util import DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, inner_join, outer_join, list_get_dict
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import DATASET_DIR
from recon.label_util import apply_label_mask, eod_fct, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct


def prep_set(dataset_dict, join_on=['root'], join_method=inner_join, asset_filter=None):
	datasets = {}

	for dataset, au_list in dataset_dict.items():
		datasets[dataset] = {}

		if (len(au_list) == 1):
			au_dg, au_cs = list_get_dict(dg, au_list[0]), list_get_dict(cs2, au_list[0])
			paths, recs, dfs = DataAPI.lazy_load(au_dg, au_cs)

			if (asset_filter is not None):
				paths = list(filter(lambda p: p[0]==asset_filter, paths))

			datasets[dataset]['paths'] = paths
			datasets[dataset]['recs'] = recs
			datasets[dataset]['dfs'] = dfs

		else:
			# TODO - Define this case better in the future
			for au in au_list:
				au_dg, au_cs = list_get_dict(dg, au), list_get_dict(cs2, au)
				paths, recs, dfs = DataAPI.lazy_load(au_dg, au_cs)

				if (asset_filter is not None):
					paths = list(filter(lambda p: p[0]==asset_filter, paths))

				datasets[dataset]['paths'] = paths
				datasets[dataset]['recs'] = recs
				datasets[dataset]['dfs'] = dfs

	return datasets

def prep_labels(label_df, groups=['bool', 'int']):
	"""
	Take label df and apply masks to produce df of label series.
	"""
	gb_label_df = delayed(lambda d: d.groupby(pd.Grouper(freq=DT_CAL_DAILY_FREQ)).last())(label_df)
	label_groups = []

	if ('bool' in groups):
		eod = delayed(eod_fct)(gb_label_df).add_suffix('_eod')
		fbeod = delayed(apply_label_mask)(gb_label_df, default_fct).add_suffix('_fbeod')
		fb = delayed(apply_label_mask)(gb_label_df, fastbreak_fct).add_suffix('_fb')
		conf = delayed(apply_label_mask)(gb_label_df, confidence_fct).add_suffix('_conf')
		fbconf = delayed(apply_label_mask)(gb_label_df, fastbreak_confidence_fct).add_suffix('_fbconf')
		label_groups.extend((eod, fbeod, fb, conf, fbconf))

	if ('int' in groups):
		vel = delayed(apply_label_mask)(gb_label_df, partial(fastbreak_fct, velocity=True)).add_suffix('_vel')
		mag = delayed(apply_label_mask)(gb_label_df, partial(confidence_fct, magnitude=True)).add_suffix('_mag')
		mom = delayed(apply_label_mask)(gb_label_df, partial(fastbreak_confidence_fct, momentum=True)).add_suffix('_mom')
		label_groups.extend((vel, mag, mom))

	labels = delayed(reduce)(outer_join, label_groups)

	return labels
