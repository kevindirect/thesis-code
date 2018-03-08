# Kevin Patel

import sys
from os import sep
from os.path import isfile, getsize
import pandas as pd
from pandas.util import hash_pandas_object
from common_util import DATA_DIR, load_df, dump_df, makedir_if_not_exists, search_df, str_now, benchmark
from data.common import DR_NAME, DR_FMT, DR_COLS, DR_IDS, DR_REQ, DR_STAGE, DR_META, DR_GEN
from data.common import NAME_IDX, DIR_IDX


class DataAPI:
	"""
	The Global API used to load or dump dataframes.
	XXX - move all dataframe logic to DataRecordAPI and have DataAPI use a generic
		interface to manipulate the record.
	"""
	class DataRecordAPI:
		"""
		Global storage structure for the storage of dataframe records dumped by other stages.
		Currently only supports single threaded access.
		
			id: integer id
			name: filename of df on disk
			root: the root dependency of this df; for raw stage data this is a join group but for others it's a df name
			basis: the direct dependency (parent) of this dataframe; for raw stage data root always equals basis
			stage: the stage the data originated from
		"""
		@classmethod
		def reload_record(cls):
			cls.DATA_RECORD = load_df(DR_NAME, dir_path=DATA_DIR, data_format=DR_FMT)
			assert list(cls.DATA_RECORD.columns)==DR_COLS, 'loaded data record columns don\'t match schema'

		@classmethod
		def reset_record(cls):
			cls.DATA_RECORD = pd.DataFrame(columns=DR_COLS)

		@classmethod
		def dump_record(cls):
			dump_df(cls.DATA_RECORD, DR_NAME, dir_path=DATA_DIR, data_format=DR_FMT)

		@classmethod
		def print_record(cls):
			print(cls.DATA_RECORD)

		@classmethod
		def assert_valid_entry(cls, entry):
			"""Assert whether or not entry is a validly formatted entry to the data record."""
			# Required fields:
			assert(all((col in entry and entry[col] is not None) for col in DR_REQ))
			assert(all((col in entry and entry[col] is not None) for col in DR_STAGE if col.startswith(entry['stage'])))

			# Required omissions (autogenerated):
			assert(all(col not in entry) for col in DR_IDS)
			assert(all(col not in entry) for col in DR_GEN)

		@classmethod
		def get_id(cls, entry):
			"""Return id of matched entry in the df record, else return a new id."""
			match = search_df(cls.DATA_RECORD, entry)
			entry_id = len(cls.DATA_RECORD.index) if (match.empty) else match.values[0]
			return entry_id, match.empty

		@classmethod
		def get_name(cls, entry):
			return '_'.join([entry['root'], entry['stage'], str(entry['id'])])

		@classmethod
		def get_path(cls, entry):
			path_dir = DATA_DIR +entry['root'] +sep +entry['basis'] +sep

			# for col_name in [col for col in DR_STAGE if col.startswith(entry['stage'])]:
			# 	if (isinstance(entry[col_name], list)):
			# 		next_dir = '_'.join(sorted(entry[col_name]))
			# 	else:
			# 		next_dir = entry[col_name]
			# 	path_dir += next_dir +sep
			return path_dir


	@classmethod
	def generate(cls, search_dict, subset=None):
		"""Provide interface to get data"""
		match_ids = search_df(cls.DataRecordAPI.DATA_RECORD, search_dict)

		for entry in cls.DataRecordAPI.DATA_RECORD.loc[match_ids].itertuples():
			yield load_df(entry[NAME_IDX], dir_path=entry[DIR_IDX], subset=subset)

	@classmethod
	def dump(cls, df, entry, save=False):
		"""Provide interface to dump new data to /DATA/"""
		cls.DataRecordAPI.assert_valid_entry(entry)

		entry['id'], new_entry = cls.DataRecordAPI.get_id(entry)
		entry['name'] = cls.DataRecordAPI.get_name(entry)
		entry['dir'] = cls.DataRecordAPI.get_path(entry)

		makedir_if_not_exists(entry['dir'])
		with benchmark('', suppress=True) as b:
			entry['size'] = dump_df(df, entry['name'], dir_path=entry['dir'])
		entry['dumptime'] = round(b.time, 2)
		entry['hash'] = sum(hash_pandas_object(df))
		addition = pd.DataFrame(columns=DR_COLS, index=[entry['id']])

		if (new_entry):
			entry['created'] = str_now()
			addition.loc[entry['id']] = entry
			cls.DataRecordAPI.DATA_RECORD = pd.concat([cls.DataRecordAPI.DATA_RECORD, addition], copy=False)
		else:
			entry['modified'] = str_now()
			addition.loc[entry['id']] = entry
			cls.DataRecordAPI.DATA_RECORD.update(addition)

		if (save):
			cls.DataRecordAPI.dump_record()
	
	@classmethod
	def save(cls):
		cls.DataRecordAPI.dump_record()


# Initialization
try:
	DataAPI.DataRecordAPI.reload_record()
except FileNotFoundError as e:
	DataAPI.DataRecordAPI.reset_record()
