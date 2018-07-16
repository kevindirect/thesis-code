# Kevin Patel

import sys
import os
from os import sep
from itertools import product
import logging

from dask import delayed

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, load_json, list_get_dict, is_empty_df, search_df, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from mutate.common import default_runt_dir_name, default_trfs_dir_name
from mutate.runt_util import RUNT_FN_TRANSLATOR, RUNT_TYPE_TRANSLATOR, RUNT_FREQ_TRANSLATOR


def run_transforms(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	runt_dir_name = default_runt_dir_name
	trfs_dir_name = default_trfs_dir_name

	runt_dir = MUTATE_DIR +runt_dir_name
	trfs_dir = runt_dir +trfs_dir_name

	# Embargo 2018 data (final validation)
	date_range = {
		'id': ('lt', 2018)
	}

	trf_defaults = load_json('trf_defaults.json', dir_path=runt_dir)
	graph = load_json('graph.json', dir_path=runt_dir)
	trfs = {}

	logging.info('loading step settings...')
	for fname in os.listdir(trfs_dir):
		trf = load_json(fname, dir_path=trfs_dir)
		meta = trf['meta']
		trfs[meta['name']] = trf

	logging.info('running task graph...')
	for path_name, path in graph.items():
		for step in path:
			logging.info('step:', step)
			process_step(trfs[step], trf_defaults)


def process_step(step_info, defaults):
	meta, fn, var, src, dst = step_info['meta'], step_info['fn'], step_info['var'], step_info['src'], step_info['dst']

	# Setting transform function
	ser_fn = RUNT_FN_TRANSLATOR[fn['ser_fn']]
	rtype_fn = RUNT_TYPE_TRANSLATOR[fn['df_fn']]
	freq = RUNT_FREQ_TRANSLATOR[fn['freq']]

	# Get indices of metadata variables
	if (meta['fmt_vars'] is not None):
		desc_var_idx = [idx for idx, var_name in enumerate(var.keys()) if (var_name in meta['fmt_vars'])]
	else:
		desc_var_idx = []

	# Setting transform parameter metadata for later
	variants = {}
	if (meta['var_fmt'] == 'grid'):
		for variant in product(*var.values()):
			desc_vars = [variant[idx] for idx in desc_var_idx]
			unique_desc = meta['rec_fmt'].format(*desc_vars)
			variants[unique_desc] = variant

	# Loading input data
	if (src is None): src = defaults['src']
	src_dg, src_cs = list_get_dict(dg, src), list_get_dict(cs2, src)
	src_paths, src_recs, src_dfs = DataAPI.load_from_dg(src_dg, src_cs)

	# Run transforms on inputs
	for key_chain in src_paths:
		logging.debug('asset: ' +key_chain[0])
		src_rec = list_get_dict(src_recs, key_chain)
		src_df = list_get_dict(src_dfs, key_chain)
		src_df = src_df.loc[search_df(src_df, date_range), :]

		for desc, variant in variants.items():
			runted_df = rtype_fn(src_df, ser_fn(variant), freq)

			if (meta['mtype_from']=='name'):       mutate_type = meta['name']
			elif (meta['mtype_from']=='rec_fmt'):  mutate_type = desc

			assert(not is_empty_df(runted_df))
			entry = make_runt_entry(desc, None, mutate_type, src_rec)
			logging.debug('dumping ' +desc +'...')
			# DataAPI.dump(runted_df, entry)


def make_runt_entry(desc, mutate_freq, mutate_type, base_rec):
	prev_hist = '' if isinstance(base_rec.hist, float) else str(base_rec.hist)

	return {
		'freq': mutate_freq,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': mutate_type,
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([prev_hist, str('mutate_' +mutate_type)]),
		'desc': desc
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		run_transforms(sys.argv[1:])
