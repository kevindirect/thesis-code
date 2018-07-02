# Kevin Patel

import sys
import os
import getopt
from itertools import product
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, get_custom_biz_freq, outer_join, right_join, wrap_parens, build_query, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import default_num_sym, default_max_seg
from mutate.pattern_util import sax_df

# TODO - Known Issue: some thresh group data is lost after saxify (rows that are not non-null in all columns)

def saxify(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	usage = lambda: print('search_hyperspace.py [-n <num_sym> -s <max_seg>]')
	num_sym = default_num_sym
	max_seg = default_max_seg
	raw_only = False

	try:
		opts, args = getopt.getopt(argv, 'hn:s:r', ['help', 'num_sym=', 'max_seg=', 'raw_only'])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-n', '--num_sym'):
			num_sym = int(arg)
		elif opt in ('-s', '--max_seg'):
			max_seg = int(arg)
		elif opt in ('-r', '--raw_only'):
			raw_only = True

	date_range = {
		'id': ('lt', 2018)
	}
	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'normalize',
		'raw_cat': 'us_equity_index'
	}
	search_query = build_query(search_terms)

	if (raw_only):
		raw_subqueries = [build_query({ 'desc': '_'.join(['raw', terms[0], terms[1]]) })
			for terms in product(['pba', 'vol'], ['dzn', 'dmx'])]
		raw_query = wrap_parens(' or '.join(raw_subqueries))
		search_query = ' and '.join([search_query, raw_query])

	norm_recs, norm_dfs = defaultdict(dict), defaultdict(dict)
	for rec, norm_df in DataAPI.generate(search_query, direct_query=True):
		norm_dfs[rec.root][rec.desc] = norm_df.loc[search_df(norm_df, date_range)]
		norm_recs[rec.root][rec.desc] = rec
	logging.info('normalize data loaded')

	for root_name in norm_recs:
		logging.info('asset: ' +str(root_name))

		for norm_type in filter(lambda n: n[-3:]=='dzn', norm_dfs[root_name]):
			logging.info('normalize type: ' +str(norm_type))
			norm_df = norm_dfs[root_name][norm_type]
			saxed_df = sax_df(norm_df, num_sym, max_seg=max_seg)
			desc = 'sax' +'(' +str(num_sym) +',' +str(max_seg) +')'
			sax_entry = make_sax_entry(desc, str('mutate_' +desc), norm_recs[root_name][norm_type])
			DataAPI.dump(saxed_df, sax_entry)

		DataAPI.update_record()

	# Normed -> SAX

	# Normed -> 1d-SAX

	# Normed -> PIP -> SAX

	# Normed -> PAA -> SAX

	# Normed -> PIP -> 1d-SAX

	# Normed -> PAA -> 1d-SAX


	

def make_sax_entry(desc, hist, base_rec):
	return {
		'freq': DT_BIZ_DAILY_FREQ,
		'root': base_rec.root,
		'basis': base_rec.name,
		'stage': 'mutate',
		'mutate_type': 'sax',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([str(base_rec.hist), hist]),
		'desc': '_'.join([base_rec.desc, desc])
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		saxify(sys.argv[1:])
