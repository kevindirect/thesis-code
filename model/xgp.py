"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import exists, basename
import logging

import numpy as np
import pandas as pd

from common_util import JSON_SFX_LEN, makedir_if_not_exists, get_cmd_args, str_to_list, is_type, is_ser, is_valid, load_json, dump_json, dump_df, benchmark
from model.common import XG_PROCESS_DIR, XG_DATA_DIR
from model.dataprep_util import COMMON_PREP_MAPPING, DATA_PREP_MAPPING
from model.datagen_util import process_group
from data.data_api import DataAPI
from recon.dataset_util import prep_dataset


def xgp(argv):
	cmd_arg_list = ['assets=', 'force']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	assets = str_to_list(cmd_input['assets=']) if (is_valid(cmd_input['assets='])) else None
	process_all = is_valid(cmd_input['force'])

	for parent_path, dirs, files in os.walk(XG_PROCESS_DIR, topdown=False):
		if (len(dirs)==0 and len(files)>0):
			group_type = basename(parent_path)
			logging.info('group {}'.format(group_type))
			for xg_fname in files:
				xg_outdir = XG_DATA_DIR +sep.join([group_type, xg_fname[:-JSON_SFX_LEN]]) +sep
				if (exists(xg_outdir) and exists(sep.join([xg_outdir, 'index.json'])) and not process_all):
					logging.info('skipping {}...'.format(xg_fname))
					continue
				logging.info('processing {}...'.format(xg_fname))
				makedir_if_not_exists(xg_outdir)
				xg_dict = load_json(xg_fname, dir_path=parent_path)
				dataset_dict = load_json(xg_dict['dataset'], dir_path=parent_path) if (is_type(xg_dict['dataset'], str)) else xg_dict['dataset']
				dataset = prep_dataset(dataset_dict, assets=assets)

				for group in xg_dict['how']:
					logging.debug('group: {}'.format(str(group)))
					assert(group_type in group)
					prep = xg_dict['prep_fn'][group_type]
					constraint = xg_dict['constraint']
					proc_paths = []
					try:
						for i, (proc_path, proc_df) in enumerate(process_group(dataset, group=group, prep=prep, constraint=constraint, delayed=False)):
							proc_paths.append(proc_path)
							logging.info('dumping {}...'.format(str(i)))
							dump_df(proc_df.to_frame() if (is_ser(proc_df)) else proc_df, str(i), dir_path=xg_outdir, data_format='pickle')
					except Exception as e:
						logging.error('exception during {}: {}'.format(str(i+1), e))
					finally:
						dump_json(proc_paths, 'index.json', dir_path=xg_outdir)


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		with DataAPI(async_writes=False):
			xgp(sys.argv[1:])
