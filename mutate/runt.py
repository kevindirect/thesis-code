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
from os.path import abspath, basename, dirname, isfile
import logging

#from dask.distributed import Client
#from multiprocessing import Pool

from common_util import MUTATE_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, NestedDefaultDict, load_json, dump_json, get_cmd_args, is_valid, isnt, is_type, get_variants, best_match, remove_dups_list, list_get_dict, is_empty_df, search_df, str_now_dtz, last_commit_dtz, benchmark
from mutate.common import HISTORY_DIR, TRANSFORMS_DIR, get_graphs, get_transforms
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
	cmd_arg_list = ['graphs=', 'force']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__), script_pkg=basename(abspath(dirname(__file__))))
	whitelist = cmd_input['graphs='].split(',') if (is_valid(cmd_input['graphs='])) else None
	runt_all = is_valid(cmd_input['force'])

	logging.info('loading settings...')
	graphs, transforms = get_graphs(whitelist=whitelist), get_transforms()

	for graph_name, graph in graphs.items():
		logging.info('running graph {}...'.format(graph_name))
		for path_name, path in graph.items():
			dep_out_of_date=False
			for level in path:
				for t in level:
					ran = [r for r in safe_process_transform((t, transforms[t]), dep_ood=dep_out_of_date, force=runt_all)]
				dep_out_of_date = any(ran)


def safe_process_transform(trf, dep_ood=False, force=False):
	"""
	Wrapper around process_transform that handles errors and history updates.

	Args:
		trf (tuple): tuple of (name, info) for transform
		dep_ood (bool): whether dependency of this transform is out of date
		force (bool): force a transform even if history exists and last update isn't out of date

	Returns:
		True if the transform ran, False otherwise
	"""
	try:
		fname, info = str(trf[0] if (trf[0].endswith('.json')) else trf[0] +'.json'), trf[1]
		hist = load_json(fname, dir_path=HISTORY_DIR) if (isfile(HISTORY_DIR +fname)) else []
		if (force or dep_ood or transform_out_of_date(fname, hist)):
			logging.info('starting {}...'.format(fname))
			process_transform(info, dump=True)
			hist.append(str_now_dtz())
			logging.info('updating history {}...'.format(fname))
			dump_json(hist, fname, dir_path=HISTORY_DIR)
			return True
		else:
			logging.info('skipping {}...'.format(fname))
			return False
	except RUNTFormatError as erf:
		error_msg = 'runt formatting error with {}: {}'.format(fname, erf)
		logging.error(error_msg)
		raise erf
	except RUNTComputeError as erc:
		error_msg = 'runt runtime error with {}: {}'.format(fname, erc)
		logging.error(error_msg)
		raise erc
	except Exception as e:
		error_msg = 'non-runt error with {}: {}'.format(fname, e)
		logging.error(error_msg)
		raise e

def transform_out_of_date(fname, hist):
	"""
	Returns True if the transform is out-of-date.
	Does not check if any dependencies are out-of-date.

	Args:
		fname (str): filename of transform info file
		hist (list): list of string date times in format '%Y-%m-%d %H:%M:%S %Z'

	Returns:
		True if transform is out-of-date (based on last git commit), false otherwise
	"""
	if (len(hist)==0):
		return True
	else:
		lc = last_commit_dtz(TRANSFORMS_DIR +fname)
		same_tz = lc.split(' ')[-1] == hist[-1].split(' ')[-1]
		return same_tz and lc > hist[-1]

def process_transform(info, dump=True):
	"""
	Process a transform.

	Args:
		info (dict): dictionary specifying the transform
		dump (bool): whether to return data, or dump data

	Returns:
		(NDD, NDD) of data or None
	"""
	if (not dump):
		out_rcs, out_dfs = NestedDefaultDict(), NestedDefaultDict()
	meta, fn, axe, var = info['meta'], info['fn'], info['axe'], info['var']
	if (meta['var_fmt']=='list' and any([s in meta['rec_fmt'] for s in ('{', '}')])):
		error_msg = 'cannot mix var_fmt==\'list\' with a parameter-inputed rec_fmt'
		logging.error(error_msg)
		raise RUNTFormatError(error_msg)

	# Loading transform, apply, and frequency settings
	rtype_fn = RUNT_TYPE_MAPPING.get(fn['df_fn'])
	ser_fn_str, col_fn_str = fn['ser_fn'], fn['col_fn']
	freq, res_freq = fn['freq'], meta['res_freq']

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
			error_msg = 'no data matched the given axefiles \'{}\' in the data_record'.format(str(src))
			logging.error(error_msg)
			raise RUNTFormatError(error_msg)

		# Run transforms on inputs
		for keychain in src_dfs:
			logging.info('data: {}'.format(str(keychain)))
			src_rc, src_df = src_rcs[keychain], src_dfs[keychain].dropna(axis=0, how='all')

			# Masking rows in src from row_mask
			if (is_valid(rm_src)):
				rm_keychain = get_rm_keychain(keychain, rm_dfs.keys())
				src_df = apply_rm(src_df, rm_dfs[rm_keychain].dropna())

			logging.debug('pre-runt: {}'.format(str(src_df)))
			if (is_empty_df(src_df)):
				error_msg = "Source DataFrame is empty (error in axefile '{}'?)".format(src)
				logging.error(error_msg)
				raise RUNTFormatError(error_msg)

			# Running variants of the transform (different sets of parameters)
			for variant in variants:
				logging.debug('variant: {}'.format(variant))
				runted_df = rtype_fn(src_df, variant, freq, ser_fn_str, col_fn_str)
				mutate_type = type_str.format(**variant) if (not meta['var_fmt']=='list') else type_str
				mutate_desc = '_'.join([keychain[-1], mutate_type])
				logging.debug('mutate_type: {}'.format(mutate_type))

				logging.debug('post-runt: {}'.format(str(runted_df)))
				if (is_empty_df(runted_df)):
					error_msg = "Result of transform is an empty DataFrame (variant: '{}')".format(str(variant))
					logging.error(error_msg)
					raise RUNTComputeError(error_msg)

				entry = make_entry('mutate', mutate_type, mutate_desc, res_freq, base_rec=src_rc)
				if (dump):
					DataAPI.dump(entry, runted_df)
				else:
					kcv = keychain + [str(variant)]
					out_rcs[kcv], out_dfs[kcv] = entry, runted_df
	if (dump):
		out = None
	else:
		out = (out_rcs, out_dfs)
	return out

def get_rm_keychain(kc, rm_kcs):
	"""
	Find and return the keychain the keychain in rm_kcs that matches kc according to some rules.

	For a match to occur, the following must be the case:
		- The first items in both lists must be identical
			* This means the root field in the data record (ie the asset) of both src and row mask are identical
		- The last item in the src keychain starts with the last item in row mask keychain
			* This means the desc or subset of the src keychain matches the desc of the row mask keychain

	Args:
		kc (list): src keychain
		rm_kc(list): row mask keychain

	Returns:
		matched keychain
	"""
	rm_match = list(filter(lambda rm_kc: kc[0]==rm_kc[0] and kc[-1].startswith(rm_kc[-1]), rm_kcs))
	logging.debug('rm_match: {}'.format(str(rm_match)))
	if (len(rm_match)!=1):
		error_msg = 'matched {} row_mask df(s), should be 1'.format(len(rm_match))
		logging.error(error_msg)
		raise RUNTComputeError(error_msg)
	return rm_match[0]

def apply_rm(data_df, rm_df):
	"""
	Apply row mask to data df.

	Args:
		data_df (pd.DataFrame): data
		rm_df (pd.DataFrame): row mask

	Returns:
		row masked df
	"""
	rm_src_diff = rm_df.index.difference(data_df.index)
	if (len(rm_src_diff)>0):
		logging.debug('rm_df.index - data_df.index: {}'.format(str(rm_src_diff)))
		res_df = data_df.loc[data_df.index & rm_df.index, :].dropna(axis=0, how='all')
	else:
		res_df = data_df.loc[rm_df.index, :].dropna(axis=0, how='all')
	return res_df

if __name__ == '__main__':
	with benchmark('ttf') as b:
		with DataAPI(async_writes=True):
			run_transforms(sys.argv[1:])
