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

from common_util import JSON_SFX_LEN, NestedDefaultDict, load_json, load_df, benchmark
from model.common import XG_PROCESS_DIR, XG_DATA_DIR
from recon.dataset_util import gen_group


def xgload(xg_subset_dir):
	ndd = NestedDefaultDict()
	for d in os.listdir(xg_subset_dir):
		ddir = xg_subset_dir +d +sep
		try:
			index = load_json('index.json', dir_path=ddir)
		except FileNotFoundError as f:
			logging.warning('index.json not found for {}, skipping...'.format(d))
			continue
		else:
			for i, path in enumerate(index):
				ndd[path] = load_df('{}.pickle'.format(i), dir_path=ddir, data_format='pickle')
	return ndd

