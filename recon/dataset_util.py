"""
Kevin Patel
"""
import sys
import os
import logging
from functools import reduce
from itertools import product

import numpy as np
import pandas as pd

from common_util import DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, isnt, is_valid, NestedDefaultDict, load_json, list_get_dict, list_set_dict, remove_dups_list
from data.data_api import DataAPI
from recon.common import DATASET_DIR


no_constraint = lambda *a: True
asset_match = lambda a, b: a[0]==b[0]
src_match = lambda a, b: a[2]==b[2]
parent_match = lambda a, b: a[-1].split('_')[:-1]==b[-1].split('_')[:-1] # Common parent if all the items in desc except last are identical

flr_asset_match = lambda fp, lp, rp: asset_match(fp, lp) and asset_match(fp, rp)
flr_src_match = lambda fp, lp, rp: src_match(fp, rp)
flr_constraint = lambda fp, lp, rp: flr_asset_match(fp, lp, rp) and flr_src_match(fp, lp, rp)

flt_parent_match = lambda fp, lp, tp: parent_match(lp, tp)
flt_asset_match = lambda fp, lp, tp: asset_match(fp, lp) and asset_match(fp, tp)
flt_constraint = lambda fp, lp, tp: flt_parent_match(fp, lp, tp) and flt_asset_match(fp, lp, tp)

fltr_parent_match = lambda fp, lp, tp, rp: parent_match(lp, tp)
fltr_asset_match = lambda fp, lp, tp, rp: asset_match(fp, lp) and asset_match(fp, tp) and asset_match(fp, rp)
fltr_src_match = lambda fp, lp, tp, rp: src_match(fp, rp)
fltr_constraint = lambda fp, lp, tp, rp: fltr_parent_match(fp, lp, tp, rp) and fltr_asset_match(fp, lp, tp, rp) and fltr_src_match(fp, lp, tp, rp)

GEN_GROUP_CONSTRAINTS = {
	'flr_constraint': flr_constraint,
	'flt_constraint': flt_constraint,
	'fltr_constraint': fltr_constraint
}

def gen_group(dataset, group=['features', 'labels', 'targets', 'row_masks'], out=['dfs'], constraint=fltr_constraint):
	"""
	Convenience function to yield specified partitions from dataset.

	Args:
		dataset (dict): dictionary returned by prep_dataset
		group (list): data partitions to include in generator
		out (list): what the generator will output, 'dfs' and/or 'recs'
		constraint (lambda, optional): constraint of data partition paths, must have as many arguments as items in group
			(if None is passed, no constraint is used and full cartesian product of groups are yielded)

	Yields:
		Pair of paths and outputs ordered by the specifed group list

		Example:
			gen_group(dataset, group=['features', 'labels'], constraint=asset_match) -> yields feature label pairs where the first items of their paths match
	"""
	if (isnt(constraint)):
		constraint = no_constraint

	parts = [list(dataset[part][out[0]].keys()) for part in group] # Can use out[0] becuase keys are guaranteed to be identical aross all outputs

	for paths in filter(lambda combo: constraint(*combo), product(*parts)):
		outputs = tuple(tuple(dataset[part][output][paths[i]] for i, part in enumerate(group)) for output in out)
		yield (paths, *outputs)

def prep_dataset(dataset_dict, assets=None, filters_map=None, dataset_dir=DATASET_DIR):
	"""
	Return the tree of lazy loaded data specified by dataset_dict, asset list, and filter mapping.
	"""
	dataset = {}
	if (isinstance(dataset_dict, str)):
		dataset_dict = load_json(dataset_dict, dir_path=dataset_dir)

	for partition, accessors in dataset_dict.items():
		if (isinstance(accessors, str)): # Dataset entry depends on another dataset, key in this and other must match
			accessors = load_json(accessors, dir_path=dataset_dir)[partition]
		filters = filters_map[partition] if (is_valid(filters_map) and partition in filters_map) else None
		dataset[partition] = prep_data(accessors, assets=assets, filters=filters)

	return dataset

def prep_data(accessors, assets=None, filters=None):
	recs, dfs = NestedDefaultDict(), NestedDefaultDict()
	flt = lambda p: (isnt(assets) or p[0] in assets) and filters_match(p, filter_lists=filters)
	for axe in accessors:
		for path, rc, df in DataAPI.axe_yield(axe, flt=flt, lazy=True):
			recs[path], dfs[path] = rc, df
	#recs, dfs = reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]), [DataAPI.axe_load(axe, flt=flt, lazy=True) for axe in accessors])
	return {'recs': recs, 'dfs': dfs}

def filters_match(item, filter_lists=None):
	def filter_match(item, filter_list):
		for idx, filter_field in enumerate(filter_list):
			if (is_valid(filter_field) and item[idx]!=filter_field):
				return False
		else:
			return True
	return True if (isnt(filter_lists)) else any(filter_match(item, flt) for flt in filter_lists)
