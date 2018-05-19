# Kevin Patel

import sys
import os
import logging
import random

import numpy as np
import pandas as pd
from numba import jit, vectorize

from common_util import search_df, get_subset, benchmark
from data.data_api import DataAPI
from recon.common import dum


def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	# Stuff
	search_terms = {
		'root': 'sp_500',
		'stage': 'mutate',
		'mutate_type': 'label'
	}
	dfs, recs = {}, {}
	for rec, label_df in DataAPI.generate(search_terms):
		recs[rec.desc] = rec
		dfs[rec.desc] = label_df
	logging.info('labels loaded')
	
	print(dfs.keys())
	print()

	rand = random.choice(dfs.keys())
	print(rand)
	print(dfs[rand])
	
if __name__ == '__main__':
	test(sys.argv[1:])