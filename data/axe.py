#    _|_|    _|      _|  _|_|_|_|
#  _|    _|    _|  _|    _|
#  _|_|_|_|      _|      _|_|_|
#  _|    _|    _|  _|    _|
#  _|    _|  _|      _|  _|_|_|_|
# JSON based data access utility.
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

from common_util import JSON_SFX_LEN, load_json, chained_filter, NestedDefaultDict
from data.common import AXEFILES_DIR, DG_PFX, CS_PFX


# ********** AXE DF GETTER **********
df_getters = {}
df_getters_file_qualifier = [{"startswith": [DG_PFX]}, {"endswith": [".json"]}]

for g in os.walk(AXEFILES_DIR, topdown=True):
	found = chained_filter(g[2], [df_getters_file_qualifier])

	if (found):
		df_getters[basename(g[0])] = {fname[len(DG_PFX):-JSON_SFX_LEN]: load_json(fname, dir_path=g[0] +sep) for fname in found}


# ********** AXE COL SUBSETTER **********
col_subsetters = {}
col_subsetters_file_qualifier = [{"startswith": [CS_PFX]}, {"endswith": [".json"]}]

for g in os.walk(AXEFILES_DIR, topdown=True):
	found = chained_filter(g[2], [col_subsetters_file_qualifier])

	if (found):
		col_subsetters[basename(g[0])] = {fname[len(CS_PFX):-JSON_SFX_LEN]: load_json(fname, dir_path=g[0] +sep) for fname in found}

col_subsetters2 = col_subsetters  # For backcompat


# ********** AXE EXCEPTION CLASSES **********
class AxeFormatError(Exception):
	"""
	Use this class for formatting/validation errors in axefiles.
	"""
	def __init__(self, message):
		super().__init__(message)


# ********** AXE UTILITIES **********
def axe_convert(axe_dg, axe_cs):
	"""
	Converts an axefile into intermediate search dict(s) used for querying the data record,
	does some validation on the axefile in the process.
	TODO
	Args:
		axe_dg (dict): axe df getter
		axe_cs (dict): axe col subsetter

	Returns:
		search dict
	"""
	if ('all' not in axe_dg and 'subsets' not in axe_dg):
		raise AxeFormatError('must have \'subsets\' and/or \'all\' field(s) in axefile')
	sd = NestedDefaultDict()

	if ('all' in dg_keys):
		sd = {'all': (**axe_dg['all'])}
		pass

	if ('subsets' in dg_keys):
		pass
	else:
	    pass

