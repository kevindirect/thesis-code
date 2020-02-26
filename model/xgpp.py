"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import exists, basename
from functools import partial
import logging

import numpy as np
import pandas as pd
import dask
from multiprocessing.pool import ThreadPool

from common_util import JSON_SFX_LEN, makedir_if_not_exists, get_cmd_args, str_to_list, is_type, is_ser, is_valid, dcompose, load_json, dump_json, dump_df, benchmark
from model.common import XG_PROCESS_DIR, XG_DATA_DIR, XVIZ_DIR
from model.dataprep_util import COMMON_PREP_MAPPING, DATA_PREP_MAPPING
from model.datagen_util import process_group
from data.data_api import DataAPI
from recon.dataset_util import prep_dataset


def xgpp(argv):
	cmd_arg_list = ['modes=', 'assets=', 'force']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	modes = str_to_list(cmd_input['modes=']) if (is_valid(cmd_input['modes='])) else ('run',)
	assets = str_to_list(cmd_input['assets=']) if (is_valid(cmd_input['assets='])) else None
	process_all = is_valid(cmd_input['force'])
	xgs, ns = [], []

	for parent_path, dirs, files in os.walk(XG_PROCESS_DIR, topdown=False):
		if (len(dirs)==0 and len(files)>0):
			group_type = basename(parent_path)
			logging.info('group {}'.format(group_type))
			for xg_fname in files:
				xg_outdir = XG_DATA_DIR +sep.join([group_type, xg_fname[:-JSON_SFX_LEN]]) +sep
				if ((not exists(xg_outdir) or not exists(sep.join([xg_outdir, 'index.json']))) or process_all):
					logging.info('queueing {}...'.format(xg_fname))
					ns.append(xg_fname)
					xg_path = sep.join([parent_path, xg_fname])
					xgs.extend(xg_process_delayed(xg_path, xg_outdir, group_type, assets))
				else:
					logging.debug('skipping {}...'.format(xg_fname))
					continue
	if ('viz' in modes):
		# Dump dask graph pictures
		assert(len(xgs)==len(ns))
		for i, xg in enumerate(xgs):
			fname = '{}_{}.svg'.format(str(i), ns[i][:-JSON_SFX_LEN])
			xgs[i].visualize(filename=XVIZ_DIR+fname)
	if ('test' in modes):
		# Test various parameters to dask.compute(...)
		res = dask.compute(*xgs) # Process experiment groups concurrently
	#if ('run' in modes):
		# Standard run
		#xgs = dask.compute(*xgs, scheduler='threads', pool=ThreadPool(len(xgs)+2)) # Process experiment groups concurrently

@dask.delayed
def lazy_dump_result(result, xg_outdir):
	logging.info('dumping {}...'.format(str(i)))
	dump_df(result[1].to_frame() if (is_ser(result[1])) else result[1], str(i), dir_path=xg_outdir, data_format='pickle')
	return result[0]

@dask.delayed
def lazy_dump_index(proc_paths, xg_outdir):
	logging.info('dumping index at {}...'.format(xg_outdir))
	proc_paths_list = list(dask.compute(*proc_paths))
	return dump_json(proc_paths_list, fname='index.json', dir_path=xg_outdir)

def xg_process_delayed(xg_path, xg_outdir, group_type, assets):
	"""
	Process experiment group. Return a list of delayed objects representing the computation.
	"""
	makedir_if_not_exists(xg_outdir)
	xg_dict = load_json(xg_path)
	dataset_dict = load_json(xg_dict['dataset'], dir_path=xg_dir) if (is_type(xg_dict['dataset'], str)) else xg_dict['dataset']
	dataset = prep_dataset(dataset_dict, assets=assets)
	xg_index_dumps = []

	for group in xg_dict['how']:
		logging.debug('group: {}'.format(str(group)))
		assert(group_type in group)
		prep = xg_dict['prep_fn'][group_type]
		constraint = xg_dict['constraint']
		proc_paths = [lazy_dump_result(d, xg_outdir)
			for i, d in enumerate(process_group(dataset, group=group, prep=prep, constraint=constraint, delayed=True))]
		xg_index_dumps.append(lazy_dump_index(proc_paths, xg_outdir))

	return xg_index_dumps


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		with DataAPI(async_writes=False):
			xgpp(sys.argv[1:])
