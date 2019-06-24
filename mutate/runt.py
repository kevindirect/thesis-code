#                       __
#     _______  ______  / /_
#    / ___/ / / / __ \/ __/
#   / /  / /_/ / / / / /_
#  /_/   \__,_/_/ /_/\__/
# run transforms module.
"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import basename, isfile
import logging

#from dask.distributed import Client

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, load_json, dump_json, get_cmd_args, is_valid, isnt, is_type, get_variants, best_match, remove_dups_list, list_get_dict, is_empty_df, search_df, str_now, benchmark
from mutate.common import RUNT_FREQ_MAPPING, HISTORY_DIR, get_graphs, get_transforms
from mutate.runt_util import RUNTFormatError, RUNTComputeError, RUNT_TYPE_MAPPING
from data.data_api import DataAPI
from data.data_util import make_entry

"""
XXX:
	* Parallelize the running of transforms (dask?)
		* If using dask:
			* Soft - (default) only compute dfs that are out of date or aren't in the data record
			* Hard - compute all dfs for each each step run (overwrite existing)
	* Log the results of this script to a file
	* Autogenerate basic access utils for all dumped data
		- Enumerate all metadata about the dumped data
"""

def run_transforms(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	cmd_arg_list = ['force']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	runt_all = is_valid(cmd_input['force'])

	logging.info('loading settings...')
	graphs, transforms = get_graphs(), get_transforms()
	#client = Client()

	for graph_name, graph in graphs.items():
		logging.info('running graph {}...'.format(graph_name))
		for path_name, path in graph.items():
			for level in path:
				process_level(level, transforms, force_level=runt_all)

def process_level(level, transforms, force_level=False):
	for t in level:
		hist = load_json(t, dir_path=HISTORY_DIR) if (isfile(HISTORY_DIR +t +'.json')) else []

		if (force_level or len(hist)==0):
			try:
				process_transform(transforms[t])
			except RUNTFormatError as erf:
				error_msg = 'runt formatting error with {}.json'.format(t)
				logging.error(error_msg)
				raise
			except RUNTComputeError as erc:
				error_msg = 'runt runtime error with {}'.format(t)
				logging.error(error_msg)
				raise
			except Exception as e:
				error_msg = 'non-runt error'
				logging.error(error_msg)
				raise
			else:
				hist.append(str_now())
				logging.info('updating history...')
				dump_json(hist, t, dir_path=HISTORY_DIR)
		else:
			logging.info('skipping {}...'.format(t))

def process_transform(info, yield_data=False):
	"""
	Process a transform.

	Args:
		info (dict): dictionary specifying the transform

	"""
	meta, fn, axe, var = info['meta'], info['fn'], info['axe'], info['var']

	if (meta['var_fmt']=='list' and any([s in meta['rec_fmt'] for s in ('{', '}')])):
		error_msg = 'cannot mix var_fmt==\'list\' with a parameter-inputed rec_fmt'
		logging.error(error_msg)
		raise RUNTFormatError(error_msg)

	# Loading transform, apply, and frequency settings
	rtype_fn = RUNT_TYPE_MAPPING.get(fn['df_fn'])
	ser_fn_str, col_fn_str = fn['ser_fn'], fn['col_fn']
	freq = RUNT_FREQ_MAPPING.get(fn['freq'], fn['freq'])
	res_freq = RUNT_FREQ_MAPPING.get(meta['res_freq'], meta['res_freq'])

	# Prep input data and row masks
	srcs = axe['src'] if (is_type(axe['src'][0], list)) else [axe['src']]
	rm_src = axe['rm']
	if (is_valid(rm_src)):
		rm_rcs, rm_dfs = DataAPI.axe_load(rm_src, lazy=False)

	# Variants and type format string
	variants = get_variants(var, meta['var_fmt'])
	type_str = meta['rec_fmt']

	for src in srcs:
		src_rcs, src_dfs = DataAPI.axe_load(src, lazy=False)

		if (len(src_dfs)==0):
			error_msg = 'no data matched the given axefiles \'{}\' in the data_record'.format(src)
			logging.error(error_msg)
			raise RUNTFormatError(error_msg)

		# Run transforms on inputs
		for keychain in src_dfs:
			logging.info('data: {}'.format(str(keychain)))
			src_rc, src_df = src_rcs[keychain], src_dfs[keychain].dropna(axis=0, how='all')

			# Masking rows in src from row_mask
			if (is_valid(rm_src)):
				rm_keychain = get_rm_keychain(keychain, rm_dfs.keys())
				rm_df = rm_dfs[rm_keychain].dropna()
				rm_src_diff = rm_df.index.difference(src_df.index)
				if (len(rm_src_diff)>0):
					logging.debug('rm_df.index - src_df.index: {}'.format(str(rm_src_diff)))
					src_df = src_df.loc[src_df.index & rm_df.index, :].dropna(axis=0, how='all')
				else:
					src_df = src_df.loc[rm_df.index, :].dropna(axis=0, how='all')

			logging.debug('pre_runt: {}'.format(str(src_df)))

			# Running variants of the transform (different sets of parameters)
			for variant in variants:
				runted_df = rtype_fn(src_df, variant, freq, ser_fn_str, col_fn_str)
				mutate_type = type_str.format(**variant)
				mutate_desc = '_'.join([keychain[-1], mutate_type])
				logging.debug('mutate_type: {}'.format(mutate_type))

				if (is_empty_df(runted_df)):
					logging.error(runted_df)
					raise RUNTComputeError('Result of transform is an empty DataFrame')

				logging.debug('post_runt: {}'.format(str(runted_df)))
				entry = make_entry('mutate', mutate_type, mutate_desc, res_freq, base_rec=src_rec)
				if (yield_data):
					yield entry, runted_df
				else:
					DataAPI.dump(entry, runted_df)


def get_rm_keychain(kc, rm_kcs):
	"""
	Find and return the keychain the keychain in rm_kcs that matches kc.
	"""
	rm_match = list(filter(lambda rm_kc: kc[-1].startswith(rm_kc[-1]), rm_kcs))
	logging.debug('rm_match: {}'.format(str(rm_match)))
	if (len(rm_match)!=1):
		error_msg = 'matched {} row_mask df(s), should be 1'.format(len(rm_match))
		logging.error(error_msg)
		raise RUNTComputeError(error_msg)
	return rm_match[0]


if __name__ == '__main__':
	with benchmark('ttf') as b:
		with DataAPI(async_writes=True):
			run_transforms(sys.argv[1:])
