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
asset_match = lambda a, b: a[0]==b[0]			# sp_500, russell_2000, dow_jones, etc.
src_match = lambda a, b: a[2].startswith(b[2])		# pba, vol, trmi2, trmi3, etc.
parent_match = lambda a, b: a[-1].split('_')[:-1]==b[-1].split('_')[:-1] # Common parent if all the items in desc except last are identical

fr_constraint = lambda f, r: asset_match(f, r) and src_match(f, r)
lt_constraint = lambda l, t: asset_match(l, t) and parent_match(l, t)

flr_constraint = lambda f, l, r: asset_match(f, l) and fr_constraint(f, r)
flt_constraint = lambda f, l, t: asset_match(f, l) and lt_constraint(l, t)

fltr_constraint = lambda f, l, t, r: asset_match(f, l) and fr_constraint(f, r) and lt_constraint(l, t)

GEN_GROUP_CONSTRAINTS = {
	'fr_constraint': fr_constraint,
	'lt_constraint': lt_constraint,
	'flr_constraint': flr_constraint,
	'flt_constraint': flt_constraint,
	'fltr_constraint': fltr_constraint,
	'asset_match': asset_match,
	None: no_constraint
}

def gen_group(dataset, group=['features', 'labels', 'targets', 'row_masks'], out=['dfs'], constraint='fltr_constraint'):
	"""
	Convenience function to yield specified partitions from dataset.

	Args:
		dataset (dict): dictionary returned by prep_dataset
		group (list): data partitions to include in generator
		out (list): what the generator will output, 'dfs' and/or 'recs'
		constraint (str, optional): string referring to data partition constraint function in GEN_GROUP_CONSTRAINTS
			must have as many arguments as items in group
			(if None is passed, no constraint is used and full cartesian product of groups are yielded)

	Yields:
		Pair of paths and outputs ordered by the specifed group list

		Example:
			gen_group(dataset, group=['features', 'labels'], constraint='asset_match') -> yields feature label pairs where the first items of their paths match
	"""
	constraint_fn = GEN_GROUP_CONSTRAINTS.get(constraint, no_constraint)
	parts = [list(dataset[part][out[0]].keys()) for part in group] # Can use out[0] becuase keys are guaranteed to be identical aross all outputs

	for paths in filter(lambda combo: constraint_fn(*combo), product(*parts)):
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
