"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import dirname, basename
import logging

import numpy as np
import pandas as pd

from common_util import JSON_SFX_LEN, load_json, chained_filter
from data.common import ACCESS_UTIL_DIR, DG_PFX, CS_PFX


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
	found = chained_filter(g[2], [df_getters_file_qualifier])

	if (found):
		df_getters[basename(g[0])] = {fname[len(DG_PFX):-JSON_SFX_LEN]: load_json(fname, dir_path=g[0] +sep) for fname in found}

"""
	********** COL SUBSETTER V2 **********
"""
col_subsetters = {}
col_subsetters_file_qualifier = {
	"exact": [],
	"startswith": [CS_PFX],
	"endswith": [".json"],
	"regex": [],
	"exclude": {
		"exact": [],
		"startswith": [DG_PFX],
		"endswith": [],
		"regex": [],
		"exclude": None
	}
}

for g in os.walk(ACCESS_UTIL_DIR, topdown=True):
	found = chained_filter(g[2], [col_subsetters_file_qualifier])

	if (found):
		col_subsetters[basename(g[0])] = {fname[len(CS_PFX):-JSON_SFX_LEN]: load_json(fname, dir_path=g[0] +sep) for fname in found}

col_subsetters2 = col_subsetters  # For backcompat

