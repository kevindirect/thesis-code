# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import chained_filter, benchmark
from data.data_api import DataAPI
from mutate.common import dum


def fix_label_dfs(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	search_terms = {
		'stage': 'mutate',
		'mutate_type': 'label'
	}

	for rec, lab_df in DataAPI.generate(search_terms):
		logging.info(rec.name)

		lab_df = fix_label_df_column_names(lab_df)
		entry = pop_gen_keys(rec)
		DataAPI.dump(lab_df, entry)

	DataAPI.update_record()

def fix_label_df_column_names(label_df):
	delimit_suffix = lambda s, l: s[:-l] +'_' +s[-l:]
	lab_suffixes = ['dir', 'mag', 'brk', 'nmb', 'nmt']
	selector = {
		"exact": [],
		"startswith": [],
		"endswith": lab_suffixes,
		"regex": [],
		"exclude": None
	}

	columns = chained_filter(label_df.columns, [selector])
	mapping = {col: delimit_suffix(col, 3) for col in columns}
	renamed = label_df.rename(mapping, axis='columns')

	return renamed


def pop_gen_keys(rec):
	entry = rec._asdict()
	allowed = ['freq', 'root', 'basis', 'stage', 'mutate_type', 'raw_cat', 'hist', 'desc']

	for key in entry.keys():
		if (key not in allowed):
			entry.pop(key, None)
	return entry


if __name__ == '__main__':
	with benchmark('ttf:') as b:
		fix_label_dfs(sys.argv[1:])