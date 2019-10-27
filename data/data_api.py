#         __      __
#    ____/ /___ _/ /_____ _
#   / __  / __ `/ __/ __ `/
#  / /_/ / /_/ / /_/ /_/ /
#  \__,_/\__,_/\__/\__,_/
# data API
"""
Kevin Patel
"""
import sys
from os import sep
from os.path import isfile, getsize
from queue import Queue, Empty
from threading import Thread
import logging

import pandas as pd
from pandas.util import hash_pandas_object
from dask import delayed

from common_util import DATA_DIR, load_df, dump_df, NestedDefaultDict, makedir_if_not_exists, chained_filter, search_df, query_df, recursive_dict, list_get_dict, list_set_dict, dict_path, isnt,  str_now, benchmark
from data.common import DR_FNAME, DR_FMT, DR_COLS, DR_IDS, DR_CAT, DR_DIR, DR_INFO, DR_GEN, DR_NAME
from data.axe import axe_get, axe_process, axe_join, axe_get_keychain


class DataAPI:
	"""
	The Global API used to load or dump dataframes. The writing is done in the inner class, the outer
	class is mostly a generic interface with convenience methods and whatnot. This is partly done to make
	swapping out the backend easier

	Potential Improvements:
		- move DataRecordAPI to it's own class file, have DataAPI inherit from it
		- implement a SQL backend for DataRecordAPI
	"""
	class DataRecordAPI:
		"""
		Synchronized connection to global storage structure for the storage of dataframe records dumped by other stages.

		Entry:
			id (int): integer id
			name (str): filename of df on disk, derived by joining fields of DR_NAME
			cat (str): data category
			root (str): the root dependency of this df
			basis (str): the direct dependency (parent) of this dataframe
			stage (str): the stage the data originated from
			type (str): type of data within the stage
			freq (str): data frequency string
			desc (str): description string
			hist (str): history string
			dir (str): directory of the serialized dataframe, derived by joining fields of DR_DIR
			size (int): size of the serialized dataframe on last write
			dumptime (str): time to dump the dataframe on last write
			hash (str): hash of the dataframe on last write
			created (str): First write (dump) time
			modified (str): Last write time, empty if it was only written once
		"""
		@classmethod
		def reload_record(cls):
			cls.DATA_RECORD = load_df(DR_FNAME, dir_path=DATA_DIR, data_format=DR_FMT)
			assert list(cls.DATA_RECORD.columns)==DR_COLS, 'loaded data record columns don\'t match schema'

		@classmethod
		def reset_record(cls):
			cls.DATA_RECORD = pd.DataFrame(columns=DR_COLS)

		@classmethod
		def start_async_handler(cls):
			"""
			Starts the asynchronous write handler.
			This is the preferred method when doing a lot of writes.
			"""
			cls.q = Queue()
			cls.alive = True
			Thread(name='DataRecordAsyncHandler', target=cls.async_handle).start()

		@classmethod
		def async_handle(cls):
			while (cls.alive):
				try:
					entry, df = cls.q.get(True, 1)
				except Empty:
					continue
				cls.dump(entry, df, dump_record=False)
				cls.q.task_done()

		@classmethod
		def async_dump(cls, entry, df):
			"""
			Enqueues a dataframe dump. This only works if start_async_handler(...) was called.
			"""
			cls.q.put((entry, df))

		@classmethod
		def kill_async_handler(cls):
			"""
			Cleans up handler and synchronizes record to disk.
			"""
			cls.q.join()
			cls.alive = False
			cls.dump_record()

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
			try:
				if (direct_query):
					match_ids = query_df(cls.DATA_RECORD, search_dict)
				else:
					match_ids = search_df(cls.DATA_RECORD, search_dict)
			except AttributeError as ae:
				logging.warning('You might not have called DataAPI.__init__() from the main thread before using DataAPI to access data')
				logging.warning('You must call __init__ for reading and use the DataAPI context manager for reading+writing')
				raise ae
			yield from cls.DATA_RECORD.loc[match_ids].itertuples()

		@classmethod
		def loader(cls, **kwargs):
			"""
			Return a loader function that takes a record entry and returns something
			"""

			def load_rec_df(rec):
				return rec, load_df(rec.name, dir_path=DATA_DIR+rec.dir, dti_freq=rec.freq, **kwargs)

			return load_rec_df

		@classmethod
		def dump(cls, entry, df, path_pfx=DATA_DIR, dump_record=False):
			"""
			Simple dataframe dump to disk for single threaded applications.
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

			if (dump_record):
				cls.dump_record()

	@classmethod
	def initialize(cls, async_writes=False):
		"""
		Initializes the static class.
		Must be called from the main thread before any work is started.
		"""
		try:
			cls.DataRecordAPI.reload_record()
		except FileNotFoundError as e:
			cls.DataRecordAPI.reset_record()
			logging.warning('Data record not found, loading empty record')
		cls.async_writes = async_writes
		if (cls.async_writes):
			cls.DataRecordAPI.start_async_handler()

	@classmethod
	def teardown(cls):
		"""
		Must call this teardown when done with all IO.
		"""
		if (cls.async_writes):
			cls.DataRecordAPI.kill_async_handler()

	@classmethod
	def __init__(cls, **kwargs):
		"""
		ContextManager init
		"""
		cls.initialize(**kwargs)

	@classmethod
	def __enter__(cls):
		"""
		ContextManager enter (doesn't do anything)
		"""
		pass

	@classmethod
	def __exit__(cls, *args):
		"""
		ContextManager exit
		"""
		cls.teardown()

	@classmethod
	def print_record(cls):
		print(cls.DataRecordAPI.get_record_view())

	@classmethod
	def generate(cls, search_dict, direct_query=False, **kwargs):
		"""
		Provide generator interface to get data.
		"""
		yield from map(cls.DataRecordAPI.loader(**kwargs), cls.DataRecordAPI.matched(search_dict, direct_query=direct_query))

	@classmethod
	def get_rec_matches(cls, search_dict, direct_query=False, **kwargs):
		"""
		Provide generator to get matched records.
		"""
		yield from cls.DataRecordAPI.matched(search_dict, direct_query=direct_query)

	@classmethod
	def get_df_from_rec(cls, rec, col_subsetter=None, path_pfx=DATA_DIR, **kwargs):
		"""
		Get dataframe from a record.
		"""
		sel = load_df(rec.name, dir_path=path_pfx+rec.dir, dti_freq=rec.freq, **kwargs)
		return sel if (isnt(col_subsetter)) else sel.loc[:, chained_filter(sel.columns, col_subsetter)]

	@classmethod
	def axe_yield(cls, axe, flt=None, lazy=False, pfx_keys=['root'], sfx_keys=['desc']):
		"""
		Yield the data returned by querying the data record with the axefile.

		Args:
			axe (list): two item list identifying the axefile
			flt (function): filter function with signature flt(path), if None no filter used
			lazy (bool): whether or not to delay the dataframe loading
			pfx_keys (list): record entry fields to prepend to each keychain
			sfx_keys (list): record entry fields to append to each keychain

		Yields:
			A tuple (keychain, record, df) consisting of:
			* keychain (list): A list corresponding to the df consisting of [pfx_keys, axe, subset, sfx_keys]
			* record (NamedTuple): The record corresponding to the dataframe in the record
			* df (pd.DataFrame|dask.delayed): df or dask.delayed df
		"""
		for sub, (df_searcher, col_subsetter) in axe_join(*map(axe_process, axe_get(axe))).items():
			for rec in cls.get_rec_matches(df_searcher):
				pfx, sfx = [getattr(rec, key) for key in pfx_keys], [getattr(rec, key) for key in sfx_keys]
				res = delayed(cls.get_df_from_rec)(rec, col_subsetter) if (lazy) else cls.get_df_from_rec(rec, col_subsetter)
				path = axe_get_keychain(pfx, axe, sfx, sub)
				if (isnt(flt) or flt(path)):
					yield path, rec, res

	@classmethod
	def axe_load(cls, axe, flt=None, lazy=False, pfx_keys=['root'], sfx_keys=['desc']):
		"""
		Return the data returned by querying the data record with the axefile as two NDDs.
		Most of the actual functionality is in axe_yield.

		Args:
			axe (list): two item list identifying the axefile
			flt (function): filter function with signature flt(path), if None no filter used
			lazy (bool): whether or not to delay the dataframe loading
			pfx_keys (list): record entry fields to prepend to each keychain
			sfx_keys (list): record entry fields to append to each keychain

		Returns:
			A tuple (recs, dfs) consisting of:
			* recs (NestedDefaultDict): a mapping of keychains to records
			* dfs (NestedDefaultDict): a mapping of keychains to dfs or dask.delayed dfs
		"""
		recs, dfs = NestedDefaultDict(), NestedDefaultDict()
		for kc, rec, df in cls.axe_yield(axe=axe, flt=flt, lazy=lazy, pfx_keys=pfx_keys, sfx_keys=sfx_keys):
			recs[kc], dfs[kc] = rec, df
		return recs, dfs

	@classmethod
	def get_record_view(cls):
		return cls.DataRecordAPI.get_record_view()

	@classmethod
	def dump(cls, entry, df, **kwargs):
		if (cls.async_writes):
			cls.DataRecordAPI.async_dump(entry, df)
		else:
			cls.DataRecordAPI.dump(entry, df, **kwargs)

	@classmethod
	def update_record(cls):
		if (cls.async_writes):
			pass
		else:
			cls.DataRecordAPI.dump_record()

