# Kevin Patel

import sys
from os import sep
import logging

from common_util import DATA_DIR, load_df, dump_df, makedir_if_not_exists, search_df, str_now, benchmark
from data.common import DR_NAME, DR_FMT, DR_COLS, DR_IDS, DR_REQ, DR_STAGE, DR_META, DR_GEN


class DataAPI:
	"""
	The Global API used to load or dump dataframes. All real implementation is done in the backend class.
	"""


	@classmethod
	def initialize(cls):
		pass
	
	@classmethod
	def print_record(cls):
		pass

	@classmethod
	def generate(cls, search_dict, **kwargs):
		"""Provide generator interface to get data"""
		pass

	@classmethod
	def dump(cls, df, entry, **kwargs):
		pass
