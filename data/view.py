# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import search_df, get_subset, benchmark
from data.common import dum
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs


def view(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	# Stuff
	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'thresh',
		'raw_cat': 'us_equity_index'
	}
	thresh_recs = {}
	thresh_dfs = {}
	for rec, thresh_df in DataAPI.generate(search_terms):
		thresh_recs[rec.root] = rec
		thresh_dfs[rec.root] = thresh_df.loc[search_df(thresh_df, date_range)]
	logging.info('thresh data loaded')

	print(thresh_dfs[thresh_dfs.keys()[0]])
	


if __name__ == '__main__':
	view(sys.argv[1:])