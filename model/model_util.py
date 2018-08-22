"""
Kevin Patel
"""

import sys
import os
import logging

import pandas as pd

from common_util import MODEL_DIR, DT_HOURLY_FREQ, DT_CAL_DAILY_FREQ, load_json, dump_json, get_cmd_args, best_match, remove_dups_list, list_get_dict, is_empty_df, search_df, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from model.common import dum


def main(argv):
	


def model_test():
	
	


if __name__ == '__main__':
	main(sys.argv[1:])