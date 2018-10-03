"""
Kevin Patel
"""

import sys
import os
import logging
from functools import partial, reduce
from itertools import product

import pandas as pd
from dask import delayed

from common_util import DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, inner_join, outer_join, list_get_dict, list_set_dict, remove_dups_list
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import DATASET_DIR
from recon.label_util import apply_label_mask, eod_fct, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct


asset_match = lambda a, b: a[0]==b[0]
src_match = lambda a, b: a[2]==b[2]

flr_asset_match = lambda fp, lp, rp: asset_match(fp, lp) and asset_match(fp, rp)
flr_src_match = lambda fp, lp, rp: src_match(fp, rp)
flr_constraint = lambda fp, lp, rp: flr_asset_match(fp, lp, rp) and flr_src_match(fp, lp, rp)

data_from_paths = lambda dataset, group, paths: tuple(list_get_dict(dataset[part]['dfs'], paths[i]) for i, part in enumerate(group))

def gen_group(dataset, group=['features', 'labels', 'row_masks'], constraint=flr_constraint):
	"""
	Convenience function to yield specified partitions from dataset.

	Args:
		dataset (dict): dictionary returned by prep_dataset
		group (list): data partitions to include in generator
		constraint (lambda): constraint of data partition paths, must have as many arguments as items in group

	Yields:
		Pair of paths and dataframes ordered by the specifed group list

		Example:
			gen_group(dataset, group=['features', 'labels'], constraint=asset_match) -> yields feature label pairs where the first items of their paths match
	"""

	parts = [dataset[part]['paths'] for part in group]
	pathgen = filter(lambda combo: constraint(*combo), product(*parts))
	datagen = map(partial(data_from_paths, dataset, group), pathgen)

	yield from zip(pathgen, datagen)

def prep_dataset(dataset_dict, assets=None, filters_map=None):
	"""
	Return the tree of lazy loaded data specified by dataset_dict, asset list, and filter mapping.
	"""
	dataset = {}

	for name, accessors in dataset_dict.items():
		filters = filters_map[name] if (filters_map is not None and name in filters_map) else None
		dataset[name] = prep_data(accessors, assets=assets, filters=filters)

	return dataset

def prep_data(accessors, assets=None, filters=None):
	au_dg, au_cs = list_get_dict(dg, accessors[0]), list_get_dict(cs2, accessors[0])
	paths, recs, dfs = DataAPI.lazy_load(au_dg, au_cs)

	if (assets is not None):
		paths = list(filter(lambda p: p[0] in assets, paths))
	if (filters is not None):
		paths = list(filter(partial(filters_match, filter_lists=filters), paths))

	data = {'paths': paths, 'recs': recs, 'dfs': dfs}

	# For data sourced from multiple data accessors
	if (len(accessors) > 1):
		for au in accessors[1:]:
			au_dg, au_cs = list_get_dict(dg, au), list_get_dict(cs2, au)
			paths, recs, dfs = DataAPI.lazy_load(au_dg, au_cs)

			if (assets is not None):
				paths = list(filter(lambda p: p[0] in assets, paths))
			if (filters is not None):
				paths = list(filter(partial(filters_match, filter_lists=filters), paths))

			data['paths'].extend(paths)
			for path in paths:
				rec, df = list_get_dict(recs, path), list_get_dict(dfs, path)
				list_set_dict(data['recs'], path, rec)
				list_set_dict(data['dfs'], path, df)

	# Assert there are no duplicates
	assert(len(remove_dups_list([''.join(lst) for lst in data['paths']]))==len(data['paths']))
	return data

def filters_match(item, filter_lists=None):
	def filter_match(item, filter_list):
		for idx, filter_field in enumerate(filter_list):
			if (filter_field is not None and item[idx]!=filter_field):
				return False
		else:
			return True

	if (filter_lists is not None):
		return any(filter_match(item, filter_list) for filter_list in filter_lists)
	else:
		return True

def prep_labels(label_df, types=['bool', 'int']):
	"""
	Take label df and apply masks to produce df of label series.
	"""
	gb_label_df = delayed(lambda d: d.groupby(pd.Grouper(freq=DT_CAL_DAILY_FREQ)).last())(label_df)
	label_groups = []

	if ('bool' in types):
		eod = delayed(eod_fct)(gb_label_df).add_suffix('_eod')
		fbeod = delayed(apply_label_mask)(gb_label_df, default_fct).add_suffix('_fbeod')
		fb = delayed(apply_label_mask)(gb_label_df, fastbreak_fct).add_suffix('_fb')
		conf = delayed(apply_label_mask)(gb_label_df, confidence_fct).add_suffix('_conf')
		fbconf = delayed(apply_label_mask)(gb_label_df, fastbreak_confidence_fct).add_suffix('_fbconf')
		label_groups.extend((eod, fbeod, fb, conf, fbconf))

	if ('int' in types):
		vel = delayed(apply_label_mask)(gb_label_df, partial(fastbreak_fct, velocity=True)).add_suffix('_vel')
		mag = delayed(apply_label_mask)(gb_label_df, partial(confidence_fct, magnitude=True)).add_suffix('_mag')
		mom = delayed(apply_label_mask)(gb_label_df, partial(fastbreak_confidence_fct, momentum=True)).add_suffix('_mom')
		label_groups.extend((vel, mag, mom))

	labels = delayed(reduce)(outer_join, label_groups)

	return labels
