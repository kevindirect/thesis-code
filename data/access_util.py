# Kevin Patel

import sys
import os
from os import sep
from os.path import dirname, basename
from functools import partial
from collections import ChainMap
import logging

import numpy as np
import pandas as pd

from common_util import JSON_SFX_LEN, load_json, chained_filter
from data.common import ACCESS_UTIL_DIR, DG_PFX, CS_PFX, default_col_subsetsfile, default_col_thresh_subsetsfile


"""
	********** DF GETTING **********
"""
df_getters = {}
df_getters_file_qualifier = {
	"exact": [],
	"startswith": [DG_PFX],
	"endswith": [".json"],
	"regex": [],
	"exclude": {
		"exact": [],
		"startswith": [CS_PFX],
		"endswith": [],
		"regex": [],
		"exclude": None
	}
}

for g in os.walk(ACCESS_UTIL_DIR, topdown=True):
	dgs = chained_filter(g[2], [df_getters_file_qualifier])

	if (dgs):
		df_getters[basename(g[0])] = {fname[len(DG_PFX):-JSON_SFX_LEN]: load_json(fname, dir_path=g[0] +sep) for fname in dgs}


"""
	********** COLUMN FILTERING **********
	
	Filter trmi etf news v2:
		trmi_etf_news_v2_qual = [col_subsetters['#trmi']['all'], col_subsetters['#trmi']['etf_filter'], col_subsetters['#trmi']['news_filter'], col_subsetters['#trmi']['v2_filter']]
		trmi_cols = chained_filter(df.columns, trmi_etf_news_v2_qual)
		df.loc[:, trmi_cols]

	Filter pba ohlc data:
		pba_ohlc_qual = [col_subsetters['#pba']['ohlc']]
		pba_cols = chained_filter(df.columns, pba_ohlc_qual)
		df.loc[:, pba_cols]

	Filter pba ohlc and vol ohlc data:
		pba_ohlc_qual = [col_subsetters['#pba']['ohlc']]
		vol_ohlc_qual = [col_subsetters['#vol']['ohlc']]
		pba_cols = chained_filter(df.columns, pba_ohlc_qual)
		vol_cols = chained_filter(df.columns, vol_ohlc_qual)
		df.loc[:, pba_cols + vol_cols]


	*********** ROW FILTERING ***********
	
	*********** TIME INDEX BASED ***********

	Filter all rows before 2018:
		date_range = {
		    'id': ('lt', 2018)
		}
		df.loc[search_df(df, date_range)]

	Filter rows to date range from 2001-02-03 to 2004-05-06
		df.loc['2001-02-03':'2004-05-06']

	Group by four year periods:
		df.groupby(pd.Grouper(freq='4Y'))

	*********** VALUE BASED ***********

	Filter all col rows in range -.1 to .2 inclusive:
		val_range = {
		    'col': ('ine', -.1, .2)
		}
		df.loc[search_df(df, val_range)]

"""

base_col_subsetsfile = default_col_subsetsfile
thresh_col_subsetsfile = default_col_thresh_subsetsfile

base_col_subsetters = load_json(base_col_subsetsfile, dir_path=ACCESS_UTIL_DIR)
thresh_col_subsetters = load_json(thresh_col_subsetsfile, dir_path=ACCESS_UTIL_DIR)

col_subsetters = ChainMap(base_col_subsetters, thresh_col_subsetters)

## COL SUBSETTER V2

col_subsetters2 = {}
col_subsetters_file_qualifier = {
	"exact": [],
	"startswith": [CS_PFX],
	"endswith": [".json"],
	"regex": [],
	"exclude": {
		"exact": [base_col_subsetsfile, thresh_col_subsetsfile],
		"startswith": [DG_PFX],
		"endswith": [],
		"regex": [],
		"exclude": None
	}
}

for g in os.walk(ACCESS_UTIL_DIR, topdown=True):
	dgs = chained_filter(g[2], [col_subsetters_file_qualifier])

	if (dgs):
		col_subsetters2[basename(g[0])] = {fname[len(CS_PFX):-JSON_SFX_LEN]: load_json(fname, dir_path=g[0] +sep) for fname in dgs}
