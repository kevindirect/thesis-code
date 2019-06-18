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

from common_util import DATA_DIR, load_df, dump_df, NestedDefaultDict, makedir_if_not_exists, chained_filter, search_df, query_df, recursive_dict, list_get_dict, list_set_dict, dict_path, isnt,  str_now, benchmark
from data.common import DR_FNAME, DR_FMT, DR_COLS, DR_IDS, DR_CAT, DR_DIR, DR_INFO, DR_GEN, DR_NAME
from data.axe import axe_get, axe_process, axe_join, axe_get_keychain


class DataAPI:
	"""
	The Global API used to load or dump dataframes. All real implementation is done in the inner class, the outer
	class is just a generic interface. This is to make swapping out the backend easier
	XXX:
		- move DataRecordAPI to it's own class file, have DataAPI inherit from it
		- implement a SQL backend for DataRecordAPI
	"""
	class DataRecordAPI:
		"""
		Global storage structure for the storage of dataframe records dumped by other stages.
		Currently only supports single threaded access.

		Entry:
			id (int): integer id
			name (str): filename of df on disk
			root (str): the root dependency of this df; for raw stage data this is a join group but for others it's a df name
			basis (str): the direct dependency (parent) of this dataframe; for raw stage data root always equals basis
			stage (str): the stage the data originated from
		"""
		@classmethod
		def reload_record(cls):
			cls.DATA_RECORD = load_df(DR_FNAME, dir_path=DATA_DIR, data_format=DR_FMT)
			assert list(cls.DATA_RECORD.columns)==DR_COLS, 'loaded data record columns don\'t match schema'

		@classmethod
		def reset_record(cls):
			cls.DATA_RECORD = pd.DataFrame(columns=DR_COLS)

		@classmethod
		def dump_record(cls):
			dump_df(cls.DATA_RECORD, DR_FNAME, dir_path=DATA_DIR, data_format=DR_FMT)

		@classmethod
		def get_record_view(cls):
			return cls.DATA_RECORD.loc[:, :]

		@classmethod
		def get_id(cls, entry):
			"""
			Return id of matched entry in the df record, else return a new id.
			"""
			match = search_df(cls.DATA_RECORD, entry)
			entry_id = len(cls.DATA_RECORD.index) if (match.empty) else match.values[0]
			return entry_id, match.empty

		@classmethod
		def get_name(cls, entry):
			"""
			Return a unique name field (used as filename of df on disk)
			"""
			return '_'.join([str(entry[field]) for field in DR_NAME])

		@classmethod
		def get_path(cls, entry):
			"""
			Return path suffix of df on disk for given candidate entry
			"""
			return sep.join([entry[field] for field in DR_DIR]) +sep

		@classmethod
		def matched(cls, search_dict, direct_query=False):
			"""
			Yield from iterator of NamedTuples from matched entry subset
			"""
			if (direct_query):
				match_ids = query_df(cls.DATA_RECORD, search_dict)
			else:
				match_ids = search_df(cls.DATA_RECORD, search_dict)
			yield from cls.DATA_RECORD.loc[match_ids].itertuples()

		@classmethod
		def loader(cls, **kwargs):
			"""Return a loader function that takes a record entry and returns something"""

			def load_rec_df(rec):
				return rec, load_df(rec.name, dir_path=DATA_DIR+rec.dir, dti_freq=rec.freq, **kwargs)

			return load_rec_df

		@classmethod
		def dump(cls, df, entry, path_pfx=DATA_DIR, update_record=False):
			"""
			XXX - break this down and make it more elegant
			"""
			entry['id'], is_new = cls.get_id(entry)
			entry['name'] = cls.get_name(entry)
			entry['dir'] = cls.get_path(entry)
			dump_location = path_pfx+entry['dir']
			makedir_if_not_exists(dump_location)
			logging.debug('dest dir: {}'.format(dump_location))

			with benchmark('', suppress=True) as b:
				entry['size'] = dump_df(df, entry['name'], dir_path=dump_location)

			entry['dumptime'] = round(b.delta.total_seconds(), 2)
			entry['hash'] = sum(hash_pandas_object(df))
			addition = pd.DataFrame(columns=DR_COLS, index=[entry['id']])

			if (is_new):
				entry['created'] = str_now()
				addition.loc[entry['id']] = entry
				cls.DATA_RECORD = pd.concat([cls.DATA_RECORD, addition], copy=False)
			else:
				entry['modified'] = str_now()
				addition.loc[entry['id']] = entry
				cls.DATA_RECORD.update(addition)

			if (update_record):
				cls.dump_record()

	@classmethod
	def initialize(cls):
		try:
			cls.DataRecordAPI.reload_record()
		except FileNotFoundError as e:
			cls.DataRecordAPI.reset_record()
			logging.warning('DataAPI initialize: Data record not found, loading empty record')

	@classmethod
	def print_record(cls):
		print(cls.DataRecordAPI.get_record_view())

	@classmethod
	def generate(cls, search_dict, direct_query=False, **kwargs):
		"""Provide generator interface to get data"""
		yield from map(cls.DataRecordAPI.loader(**kwargs), cls.DataRecordAPI.matched(search_dict, direct_query=direct_query))

	@classmethod
	def get_rec_matches(cls, search_dict, direct_query=False, **kwargs):
		"""Provide generator to get matched records"""
		yield from cls.DataRecordAPI.matched(search_dict, direct_query=direct_query)

	@classmethod
	def get_df_from_rec(cls, rec, col_subsetter=None, path_pfx=DATA_DIR, **kwargs):
		sel = load_df(rec.name, dir_path=path_pfx+rec.dir, dti_freq=rec.freq, **kwargs)
		return sel if (isnt(col_subsetter)) else sel.loc[:, chained_filter(sel.columns, col_subsetter)]

	@classmethod
	def axe_yield(cls, axe, lazy=False, pfx_keys=['root'], sfx_keys=['desc']):
		"""
		Yield the data returned by querying the data record with the axefile.

		Args:
			axe (list): two item list identifying the axefile
			lazy (bool): whether or not to delay the dataframe loading
			pfx_keys (list): record entry fields to prepend to each keychain
			sfx_keys (list): record entry fields to append to each keychain

		Yields:
			A tuple (keychain, record, df) consisting of:
			* keychain (list): A list corresponding to the df consisting of [pfx_keys, axe, subset, sfx_keys]
			* record (NamedTuple): The record corresponding to the dataframe in the record
			* df (pd.DataFrame|dask.delayed): df or dask.delayed df
		"""
		for name, (df_searcher, col_subsetter) in axe_join(*map(axe_process, axe_get(axe))).items():
			for rec in cls.get_rec_matches(df_searcher):
				pfx, sfx, sub = [getattr(rec, key) for key in pfx_keys], [getattr(rec, key) for key in sfx_keys], [name]
				res = delayed(cls.get_df_from_rec)(rec, col_subsetter) if (lazy) else cls.get_df_from_rec(rec, col_subsetter)
				yield axe_get_keychain(pfx, axe, sub, sfx), rec, res

	@classmethod
	def axe_load(cls, axe, lazy=False, pfx_keys=['root'], sfx_keys=['desc']):
		"""
		Return the data returned by querying the data record with the axefile as two NDDs.
		Most of the actual functionality is in axe_yield.

		Args:
			axe (list): two item list identifying the axefile
			lazy (bool): whether or not to delay the dataframe loading
			pfx_keys (list): record entry fields to prepend to each keychain
			sfx_keys (list): record entry fields to append to each keychain

		Returns:
			A tuple (recs, dfs) consisting of:
			* recs (NestedDefaultDict): a mapping of keychains to records
			* dfs (NestedDefaultDict): a mapping of keychains to dfs or dask.delayed dfs
		"""
		recs, dfs = NestedDefaultDict(), NestedDefaultDict()
		for kc, rec, df in axe_yield(cls, axe, lazy=lazy, pfx_keys=pfx_keys, sfx_keys=sfx_keys):
			recs[kc], dfs[kc] = rec, df
		return recs, dfs

	@classmethod
	def get_record_view(cls):
		return cls.DataRecordAPI.get_record_view()

	@classmethod
	def dump(cls, df, entry, **kwargs):
		cls.DataRecordAPI.dump(df, entry, **kwargs)

	@classmethod
	def update_record(cls):
		cls.DataRecordAPI.dump_record()


DataAPI.initialize()
