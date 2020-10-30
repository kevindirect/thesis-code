"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import basename, dirname, exists
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import logging

import numpy as np
import pandas as pd
import dask
from multiprocessing.pool import ThreadPool

from common_util import JSON_SFX_LEN, makedir_if_not_exists, get_cmd_args, str_to_list, is_type, is_ser, is_valid, dcompose, load_json, dump_json, dump_df, benchmark
from model.common import XG_PROCESS_DIR, XG_DATA_DIR, XG_INDEX_FNAME, XG_VIZ_DIR
from model.dataprep_util import COMMON_PREP_MAPPING, DATA_PREP_MAPPING
from model.datagen_util import process_group
from data.data_api import DataAPI
from recon.dataset_util import prep_dataset


def xgpp(argv):
	cmd_arg_list = ['modes=', 'assets=', 'force']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__), \
		script_pkg=basename(dirname(__file__)))
	modes = str_to_list(cmd_input['modes=']) if (is_valid(cmd_input['modes='])) \
		else ('run',) # possible modes: run, viz, test
	assets = str_to_list(cmd_input['assets=']) if (is_valid(cmd_input['assets='])) \
		else None
	process_all = is_valid(cmd_input['force'])
	xgs = []

	for parent_path, dirs, files in os.walk(XG_PROCESS_DIR, topdown=False):
		if (len(dirs)==0 and len(files)>0):
			group_type = basename(parent_path)
			logging.info('group {}'.format(group_type))
			for xg_fname in files:
				xg_name = xg_fname[:-JSON_SFX_LEN]
				xg_outdir = XG_DATA_DIR +sep.join([group_type, xg_name]) +sep
				if ((not exists(xg_outdir) or not exists(sep.join([xg_outdir, XG_INDEX_FNAME]))) or process_all):
					logging.info('queueing {}...'.format(xg_fname))
					xg_path = sep.join([parent_path, xg_fname])
					xg_objs = xg_process_delayed(xg_path, xg_outdir, group_type, assets)
					xgs.extend(xg_objs)
					if ('viz' in modes):
						logging.info('visualizing {}...'.format(xg_fname))
						assert(len(xg_objs)==1)
						viz_outdir = XG_VIZ_DIR+group_type+sep
						makedir_if_not_exists(viz_outdir)
						xg_objs[0].visualize(filename='{}{}.svg'.format(viz_outdir, xg_name))
				else:
					logging.debug('skipping {}...'.format(xg_fname))
					continue

	if ('test' in modes):
		# Test various parameters to dask.compute(...)
		times = []

		with benchmark('0') as b:
			res = dask.compute(*xgs)
		times.append(b.delta)
		with benchmark('1') as b:
			res = dask.compute(*xgs, traverse=False)
		times.append(b.delta)
		with benchmark('2') as b:
			res = dask.compute(*xgs, scheduler='synchronous')
		times.append(b.delta)

		with benchmark('3') as b:
			res = dask.compute(*xgs, scheduler='threads')
		times.append(b.delta)
		with benchmark('4') as b:
			res = dask.compute(*xgs, scheduler='threads', pool=ThreadPool(2))
		times.append(b.delta)
		with benchmark('5') as b:
			res = dask.compute(*xgs, scheduler='threads', pool=ThreadPool(4))
		times.append(b.delta)
		with benchmark('6') as b:
			res = dask.compute(*xgs, scheduler='threads', pool=ThreadPool(8))
		times.append(b.delta)
		with benchmark('7') as b:
			res = dask.compute(*xgs, scheduler='threads', pool=ThreadPool(16))
		times.append(b.delta)

		with benchmark('8') as b:
			res = dask.compute(*xgs, scheduler='processes')
		times.append(b.delta)
		with benchmark('9') as b:
			res = dask.compute(*xgs, scheduler='processes', pool=Pool(2))
		times.append(b.delta)
		with benchmark('10') as b:
			res = dask.compute(*xgs, scheduler='processes', pool=Pool(4))
		times.append(b.delta)
		with benchmark('11') as b:
			res = dask.compute(*xgs, scheduler='processes', pool=Pool(8))
		times.append(b.delta)
		with benchmark('12') as b:
			res = dask.compute(*xgs, scheduler='processes', pool=Pool(16))
		times.append(b.delta)
		with benchmark('13') as b:
			res = dask.compute(*xgs, scheduler='processes', traverse=False)
		times.append(b.delta)

		print(times)
	if ('run' in modes):
		# Standard run
		xgs = dask.compute(*xgs, scheduler='processes', traverse=False)

@dask.delayed
def lazy_dump_result(result, fname, xg_outdir):
	logging.info('dumping {}...'.format(str(fname)))
	dump_df(result[1].to_frame() if (is_ser(result[1])) else result[1], fname, dir_path=xg_outdir, data_format='pickle')
	return result[0]

@dask.delayed
def lazy_dump_index(proc_paths, xg_outdir):
	logging.info('dumping index at {}...'.format(xg_outdir))
	proc_paths_list = list(dask.compute(*proc_paths, scheduler='threads', traverse=False))
	return dump_json(proc_paths_list, fname=XG_INDEX_FNAME, dir_path=xg_outdir)

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
		proc_paths = [lazy_dump_result(d, str(i), xg_outdir)
			for i, d in enumerate(process_group(dataset, group=group, prep=prep, constraint=constraint, delayed=True))]
		xg_index_dumps.append(lazy_dump_index(proc_paths, xg_outdir))

	return xg_index_dumps


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		with DataAPI(async_writes=False):
			xgpp(sys.argv[1:])
