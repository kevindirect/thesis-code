"""
Kevin Patel
"""

import sys
import os
import logging

import pandas as pd

from common_util import MODEL_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, load_json, dump_json, get_cmd_args, best_match, remove_dups_list, list_get_dict, inner_join, is_empty_df, search_df, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import DATASET_DIR


def prep_set(dataset_dict, join_on=['root'], join_method=inner_join):
	datasets = {}

	for dataset, au_list in dataset_dict.items():
		datasets[dataset] = {}

		if (len(au_list) == 1):
			au_dg, au_cs = list_get_dict(dg, au_list[0]), list_get_dict(cs2, au_list[0])
			paths, recs, dfs = DataAPI.lazy_load(au_dg, au_cs)

			datasets[dataset]['paths'] = paths
			datasets[dataset]['recs'] = recs
			datasets[dataset]['dfs'] = dfs

		else:
			# TODO - Define this case better in the future
			for au in au_list:
				au_dg, au_cs = list_get_dict(dg, au), list_get_dict(cs2, au)
				paths, recs, dfs = DataAPI.lazy_load(au_dg, au_cs)

				datasets[dataset]['paths'] = paths
				datasets[dataset]['recs'] = recs
				datasets[dataset]['dfs'] = dfs

	return datasets
