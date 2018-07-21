# Kevin Patel

import sys
import os
from os import sep
from itertools import product
import logging

from dask import delayed

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, load_json, best_match, remove_dups_list, list_get_dict, is_empty_df, search_df, benchmark
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
		trfs[trf['meta']['name']] = trf

	logging.info('running task graph...')
	for path_name, path in graph.items():
		for step in path:
			logging.info('step: ' +str(step))
			step_info = fill_defaults(trfs[step], trf_defaults)
			process_step(step_info, date_range)

def fill_defaults(step_info, defaults):
	if (step_info['src'] is None): step_info['src'] = defaults['src']
	return step_info

def get_row_mask_keychain(original_keychain, all_mask_keys):
	"""
	Return mask keychain that best corresponds to original_keychain
	"""
	assert(len(original_keychain)==len(all_mask_keys))
	mapped = [best_match(key, all_mask_keys[idx], alt_maps={'thresh': 'raw'}) for idx, key in enumerate(original_keychain)]
	return mapped

def process_step(step_info, date_range):
	meta, fn, var, rm, src, dst = step_info['meta'], step_info['fn'], step_info['var'], step_info['rm'], step_info['src'], step_info['dst']

	# Loading transform, apply, and frequency settings
	ser_fn = RUNT_FN_TRANSLATOR[fn['ser_fn']]
	rtype_fn = RUNT_TYPE_TRANSLATOR[fn['df_fn']]
	freq = RUNT_FREQ_TRANSLATOR[fn['freq']]

	# Making all possible parameter combinations
	if (meta['var_fmt'] == 'grid'):
		var_names, param_combos = list(var.keys()), list(product(*var.values()))
		variants = [{var_names[idx]: param_value for idx, param_value in enumerate(combo)} for combo in param_combos]

	# Loading row mask, if any
	if (rm is not None):
		rm_dg, rm_cs = list_get_dict(dg, rm), list_get_dict(cs2, rm)
		rm_paths, rm_recs, rm_dfs = DataAPI.load_from_dg(rm_dg, rm_cs)
		rm_keys = [remove_dups_list([key_chain[i] for key_chain in rm_paths]) for i in range(len(rm_paths[0]))]

	# Loading input data
	src_dg, src_cs = list_get_dict(dg, src), list_get_dict(cs2, src)
	src_paths, src_recs, src_dfs = DataAPI.load_from_dg(src_dg, src_cs)

	# Run transforms on inputs
	for key_chain in src_paths:
		logging.info('data: ' +str('_'.join(key_chain)))
		src_rec, src_df = list_get_dict(src_recs, key_chain), list_get_dict(src_dfs, key_chain)
		src_df = src_df.loc[search_df(src_df, date_range), :].dropna(axis=0, how='all')
		print('before:', src_df)

		# Masking rows in src from row mask
		if (rm is not None):
			rm_key_chain = get_row_mask_keychain(key_chain, rm_keys)
			rm_df = list_get_dict(rm_dfs, rm_key_chain).dropna()
			not_in_src = rm_df.index.difference(src_df.index)
			logging.debug('row mask: ' +str('_'.join(rm_key_chain)))
			if (len(not_in_src)>0):
				logging.debug('rm_idx - src_idx: ' +str(not_in_src))
				src_df = src_df.loc[src_df.index & rm_df.index, :]
			else:
				src_df = src_df.loc[rm_df.index, :]

		print('after:', src_df)

		# Running variants of the transform
		for variant in variants:
			runted_df = rtype_fn(src_df, ser_fn(**variant), freq)
			desc = meta['rec_fmt'].format(**variant)

			if (meta['mtype_from']=='name'):       mutate_type = meta['name']
			elif (meta['mtype_from']=='rec_fmt'):  mutate_type = desc

			assert(not is_empty_df(runted_df))
			entry = make_runt_entry(desc, None, mutate_type, src_rec)
			logging.info('dumping ' +desc +'...')
			print(runted_df)
			# DataAPI.dump(runted_df, entry)
	
	# DataAPI.update_record() # Sync


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
