"""
Kevin Patel
"""
import sys
from os import sep
from os.path import isfile, getsize
from copy import deepcopy
from collections import defaultdict
import logging

from dask import delayed
import pandas as pd
from pandas.util import hash_pandas_object

from common_util import DATA_DIR, isnt, is_type
from data.common import DR_NAME, DR_FMT, DR_COLS, DR_IDS, DR_REQ, DR_META, DR_GEN


class DataRecordFormatError(Exception):
	"""
	Use this class for formatting/validation errors in producing data record entries.
	"""
	def __init__(self, message):
		super().__init__(message)


def make_entry(stage, stage_type, desc, freq, base_rec=None, name=None, cat=None):
	"""
	Make and return a valid data record entry.
	"""
	def field_check(name, var):
		"""
		Run basic checks on data record entry fields, return (name, var) if all tests pass.
		"""
		if (isnt(var)):
			raise DataRecordFormatError('{} cannot be None'.format(name))
		elif (not is_type(var, str)):
			raise DataRecordFormatError('{} must be a string'.format(name))
		elif (var==''):
			raise DataRecordFormatError('{} cannot be an empty string'.format(name))
		else:
			return (name, var)

	if (isnt(base_rec) and (isnt(name) and isnt(cat))):
		raise DataRecordFormatError('either base record or name+category must be provided')

	hist = '{s}_{t}'.format(s=stage, t=stage_type)
	if (isnt(base_rec)):
		root = name
		basis = name
		cat = cat
	else:
		root = base_rec.root
		basis = base_rec.name
		cat = base_rec.cat
		prev_hist = '' if (is_type(base_rec.hist, float)) else str(base_rec.hist)
		hist = '->'.join([prev_hist, hist]),

	return dict([
		field_check('freq', freq),
		field_check('root', root),
		field_check('basis', basis),
		field_check('stage', stage),
		field_check('type', stage_type),
		field_check('cat', cat),
		field_check('hist', hist),
		field_check('desc', desc)
	])


def validate_axefile(axe, axe_cs, axe_dg, *records):
	"""
	Validate an axefiles formatting.
	"""
#	if (axe[1]!=axe_cs.keys()[0]):
#		logging.error('cs file first key must match axefile name')

#	if (axe[1]!=axe_dg.keys()[0]):
#		logging.error('dg file first key must match axefile name')

	if (axe[0]==axe[1]):
		logging.debug('root axefile')

	elif (axe[1].startswith(axe[0])):
		logging.debug('view axefile on {}'.format(axe[0]))

	else:
		logging.error()
		raise AxeFormatError('invalid axefile name \'{}\''.format(str(axe)))


def get_effective_desc(axefile, ):
	"""
	Consistent rule for getting desc id based on axefile.
	"""
	pass
