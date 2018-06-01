# Kevin Patel

import sys
import os
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from numba import jit, vectorize

from common_util import DT_HOURLY_FREQ, DT_BIZ_DAILY_FREQ, get_custom_biz_freq, dti_to_ymd, outer_join, right_join, search_df, chained_filter, benchmark
from data.data_api import DataAPI
from data.access_util import col_subsetters as cs
from mutate.common import dum
from mutate.pattern import sax_df


def saxify(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

	date_range = {
		'id': ('lt', 2018)
	}
	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'normalize',
		'raw_cat': 'us_equity_index'
	}
	norm_recs, norm_dfs = defaultdict(dict), defaultdict(dict)
	for rec, norm_df in DataAPI.generate(search_terms):
		norm_dfs[rec.root][rec.desc] = norm_df.loc[search_df(norm_df, date_range)]
		norm_recs[rec.root][rec.desc] = rec
	logging.info('normalize data loaded')

	num_sym = 4
	max_seg = 8
	for root_name in norm_recs:
		logging.info('asset: ' +str(root_name))

		for norm_type in filter(lambda n: n[-3:]=='dzn', norm_dfs[root_name]):
			logging.info('normalize type: ' +str(norm_type))
			norm_df = norm_dfs[root_name][norm_type]
			saxed_df = dti_to_ymd(sax_df(norm_df, num_sym, max_seg=max_seg))
			desc = 'saxify' +'(' +str(num_sym) +',' +str(max_seg) +')'
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
		'mutate_type': 'saxify',
		'raw_cat': base_rec.raw_cat,
		'hist': '->'.join([str(base_rec.hist), hist]),
		'desc': '_'.join([base_rec.desc, desc])
	}


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		saxify(sys.argv[1:])
