import numpy as np
import pandas as pd
import sys
import os
import re
from os.path import dirname, basename, realpath, exists, isfile
from functools import partial, partialmethod



def io_test(argv):
	test_dir = get_script_dir() +'test' +os.sep
	test_base = 'sp_500_after'

	for ftype, ext_list in FMT_EXTS.items():
		if (ftype != 'csv'):
			fname = test_base + ftype
			print(ftype)
			df = load_df(fname, dir_path=test_dir, data_format=ftype)
			if (df is not None):
				dump_df(df, fname, dir_path=test_dir, data_format=ftype)
			else:
				print('load issue')
				sys.exit(1)

DF_DATA_FMT='parquet'

get_script_dir = lambda: dirname(realpath(sys.argv[0])) +os.sep

def _require(_item, _type):
	if (not isinstance(_item, _type)):
		raise ValueError('item must be a ', _type)

FMT_EXTS = {
	'csv': ('.csv',),
	'feather': ('.feather',),
	'hdf_fixed': ('.h5', '.hdf', '.he5', '.hdf5'),
	'hdf_table': ('.h5', '.hdf', '.he5', '.hdf5'),
	'parquet': ('.parquet',)
}

def load_df(fname, dir_path=None, subset=None, data_format=DF_DATA_FMT):
	"""Assumes that source file has a non-default index column as the first column"""
	ext_tuple = FMT_EXTS[data_format]
	fpath = str(dir_path + fname) if dir_path else fname
	if (not fname.endswith(ext_tuple)):
		fpath += ext_tuple[0]

	if (isfile(fpath)):
		try:
			load_fn = {
				'csv': partial(pd.read_csv, index_col=0, usecols=subset),
				'feather': partial(pd.read_feather),
				'hdf_fixed': partial(pd.read_hdf, key=None, mode='r', columns=subset, format='fixed'),
				'hdf_table': partial(pd.read_hdf, key=None, mode='r', columns=subset, format='table'),
				'parquet': partial(pd.read_parquet, columns=subset)
			}.get(data_format)
			df = load_fn(fpath)
			return df.set_index('id') if data_format=='feather' else df

		except Exception as e:
			print('error during load:', e)
			return None
	else:
		print(basename(fpath), 'must be in the following directory:', dirname(fpath))
		return None


def dump_df(df, fname, dir_path=None, data_format=DF_DATA_FMT):
	ext_tuple = FMT_EXTS[data_format]
	fpath = str(dir_path + fname) if dir_path else fname
	if (not fname.endswith(ext_tuple)):
		fpath += ext_tuple[0]

	try:
		{
			'csv': df.to_csv,
			'feather': (lambda f: df.reset_index().to_feather(f)),
			'hdf_fixed': partial(df.to_hdf, fname, mode='w', format='fixed'),
			'hdf_table': partial(df.to_hdf, fname, mode='w', format='table'),
			'parquet': df.to_parquet
		}.get(data_format)(fpath)
	except Exception as e:
		print('error during dump:', e)
		return None


if __name__ == '__main__':
	io_test(sys.argv[1:])
