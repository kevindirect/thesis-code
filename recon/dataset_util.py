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

from common_util import DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, load_json, best_match, df_dti_index_to_date, inner_join, outer_join
from common_util import NestedDefaultDict, list_get_dict, list_set_dict, remove_dups_list
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import DATASET_DIR

no_constraint = lambda *a: True
asset_match = lambda a, b: a[0]==b[0]
src_match = lambda a, b: a[2]==b[2]

flr_asset_match = lambda fp, lp, rp: asset_match(fp, lp) and asset_match(fp, rp)
flr_src_match = lambda fp, lp, rp: src_match(fp, rp)
flr_constraint = lambda fp, lp, rp: flr_asset_match(fp, lp, rp) and flr_src_match(fp, lp, rp)

def gen_group(dataset, group=['features', 'labels', 'row_masks'], constraint=flr_constraint):
	"""
	Convenience function to yield specified partitions from dataset.

	Args:
		dataset (dict): dictionary returned by prep_dataset
		group (list): data partitions to include in generator
		constraint (lambda, optional): constraint of data partition paths, must have as many arguments as items in group
			(if None is passed, no constraint is used and full cartesian product of groups are yielded)

	Yields:
		Pair of paths and dataframes ordered by the specifed group list

		Example:
			gen_group(dataset, group=['features', 'labels'], constraint=asset_match) -> yields feature label pairs where the first items of their paths match
	"""
	if (constraint is None):
		constraint = no_constraint

	parts = [list(dataset[part]['dfs'].keys()) for part in group]

	for paths in filter(lambda combo: constraint(*combo), product(*parts)):
		yield paths, tuple(dataset[part]['dfs'][paths[i]] for i, part in enumerate(group))

def prep_dataset(dataset_dict, assets=None, filters_map=None, dataset_dir=DATASET_DIR):
	"""
	Return the tree of lazy loaded data specified by dataset_dict, asset list, and filter mapping.
	"""
	dataset = {}

	for partition, accessors in dataset_dict.items():
		if (isinstance(accessors, str)): # Dataset entry depends on another dataset, key in this and other must match
			accessors = load_json(accessors, dir_path=dataset_dir)[partition]
		filters = filters_map[partition] if (filters_map is not None and partition in filters_map) else None
		dataset[partition] = prep_data(accessors, assets=assets, filters=filters)

	return dataset

def prep_data(accessors, assets=None, filters=None):
	recs, dfs = NestedDefaultDict(), NestedDefaultDict()

	for au in accessors:
		for path, rec, df in DataAPI.lazy_yield(list_get_dict(dg, au), list_get_dict(cs2, au)):
			if ((assets is None or path[0] in assets) and filters_match(path, filter_lists=filters)):
				recs[path], dfs[path] = rec, df

	# Assert there are no duplicate path keys
	assert(len(remove_dups_list([''.join(lst) for lst in recs.keys()]))==len(recs))
	assert(len(remove_dups_list([''.join(lst) for lst in dfs.keys()]))==len(dfs))
	return {'recs': recs, 'dfs': dfs}

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
