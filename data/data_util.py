#         __      __
#    ____/ /___ _/ /_____ _
#   / __  / __ `/ __/ __ `/
#  / /_/ / /_/ / /_/ /_/ /
#  \__,_/\__,_/\__/\__,_/
# data utilities module.
"""
Kevin Patel
"""
import sys
from os import sep
import logging

from dask import delayed
import pandas as pd

from common_util import DATA_DIR, isnt, is_type
from data.common import DR_MAN, DR_COLS


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
		hist = '->'.join([prev_hist, hist])

	entry = dict([
		field_check('cat', cat),
		field_check('root', root),
		field_check('basis', basis),
		field_check('stage', stage),
		field_check('type', stage_type),
		field_check('freq', freq),
		field_check('desc', desc),
		field_check('hist', hist)
	])

	if (set(entry.keys())!=set(DR_MAN)):
		error_msg = 'Wrong or missing fields were manually set in this entry'
		logging.error(error_msg)
		raise DataRecordFormatError(error_msg)
	return entry


