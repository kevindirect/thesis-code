
import sys
import pandas as pd
import os
from os.path import dirname, realpath
from timeit import default_timer

def main(argv):
	test_ver = '_b'
	test_dir = get_script_dir() +'test' +test_ver +os.sep
	assets = list(map(lambda s:s[:-4], os.listdir(test_dir)))
	ftypes = ['fixed_hdf', 'table_hdf', 'feather', 'parquet']
	dump_df, size_df, load_df = pd.DataFrame(columns=ftypes), pd.DataFrame(columns=ftypes), pd.DataFrame(columns=ftypes)
	print(assets)
	for f in assets:
		print(f)
		df = pd.read_csv(test_dir +f +'.csv', index_col=0)
		d, s, l = speed_test(df.loc[:, df.dtypes != object], f, test_dir, ftypes)
		dump_df.loc[f] = d
		size_df.loc[f] = s
		load_df.loc[f] = l

	dump_df.to_csv(get_script_dir()+ 'dump_df' +test_ver +'.csv')
	size_df.to_csv(get_script_dir()+ 'size_df' +test_ver +'.csv')
	load_df.to_csv(get_script_dir()+ 'load_df' +test_ver +'.csv')

	means, stds, sums = pd.DataFrame(columns=ftypes), pd.DataFrame(columns=ftypes), pd.DataFrame(columns=ftypes)
	means.loc['mean dump time'] = dump_df.mean(axis=0)
	means.loc['mean load time'] = load_df.mean(axis=0)
	means.loc['mean filesize'] = size_df.mean(axis=0)

	stds.loc['std dump time'] = dump_df.std(axis=0)
	stds.loc['std load time'] = load_df.std(axis=0)
	stds.loc['std filesize'] = size_df.std(axis=0)

	sums.loc['sum dump time'] = dump_df.sum(axis=0)
	sums.loc['sum load time'] = load_df.sum(axis=0)
	sums.loc['sum filesize'] = size_df.sum(axis=0)

	print('means')
	print(means)
	print()

	print('stds')
	print(stds)
	print()

	print('sums')
	print(sums)

	return

get_script_dir = lambda: dirname(realpath(sys.argv[0])) +os.sep

def speed_test(test_df, test_name, dir_path, ftypes):
	dumps = dict.fromkeys(ftypes)
	sizes = dict.fromkeys(ftypes)
	loads = dict.fromkeys(ftypes)

	print('********** DUMP TESTS **********')
	ftype = 'fixed_hdf'
	fpath = dir_path +test_name +ftype +'.h5'
	start = default_timer()
	test_df.to_hdf(fpath, test_name, mode='w', format='fixed')
	dumps[ftype] = default_timer() - start
	sizes[ftype] = os.path.getsize(fpath) >> 20

	ftype = 'table_hdf'
	fpath = dir_path +test_name +ftype +'.h5'
	start = default_timer()
	test_df.to_hdf(fpath, test_name, mode='w', format='table')
	dumps[ftype] = default_timer() - start
	sizes[ftype] = os.path.getsize(fpath) >> 20

	ftype = 'feather'
	fpath = dir_path +test_name +ftype +'.feather'
	start = default_timer()
	test_df.reset_index().to_feather(fpath)
	dumps[ftype] = default_timer() - start
	sizes[ftype] = os.path.getsize(fpath) >> 20

	ftype = 'parquet'
	fpath = dir_path +test_name +ftype +'.parquet'
	start = default_timer()
	test_df.to_parquet(fpath, compression='gzip')
	dumps[ftype] = default_timer() - start
	sizes[ftype] = os.path.getsize(fpath) >> 20

	ftype = 'csv'
	if (ftype in ftypes):
		fpath = dir_path +test_name +ftype +'.csv'
		start = default_timer()
		test_df.to_csv(fpath)
		dumps[ftype] = default_timer() - start
		sizes[ftype] = os.path.getsize(fpath) >> 20

	print('\n********** LOAD TESTS **********')
	ftype = 'fixed_hdf'
	fpath = dir_path +test_name +ftype +'.h5'
	start = default_timer()
	test_fixed_hdf = pd.read_hdf(fpath, test_name, mode='r', format='fixed')
	loads[ftype] = default_timer() - start

	ftype = 'table_hdf'
	fpath = dir_path +test_name +ftype +'.h5'
	start = default_timer()
	test_table_hdf = pd.read_hdf(fpath, test_name, mode='r', format='table')
	loads[ftype] = default_timer() - start

	ftype = 'feather'
	fpath = dir_path +test_name +ftype +'.feather'
	start = default_timer()
	test_feather = pd.read_feather(fpath).set_index('id')
	loads[ftype] = default_timer() - start

	ftype = 'parquet'
	fpath = dir_path +test_name +ftype +'.parquet'
	start = default_timer()
	test_parquet = pd.read_parquet(fpath)
	loads[ftype] = default_timer() - start

	ftype = 'csv'
	if (ftype in ftypes):
		fpath = dir_path +test_name +ftype +'.csv'
		start = default_timer()
		test_csv = pd.read_csv(fpath, index_col=0)
		loads[ftype] = default_timer() - start

	assert(test_df.equals(test_fixed_hdf))
	assert(test_df.equals(test_table_hdf))
	assert(test_df.equals(test_feather))
	assert(test_df.equals(test_parquet))
	# assert(test_csv.equals(test_csv))

	os.remove(dir_path +test_name +'fixed_hdf.h5')
	os.remove(dir_path +test_name +'table_hdf.h5')
	os.remove(dir_path +test_name +'feather.feather')
	os.remove(dir_path +test_name +'parquet.parquet')
	if ('csv' in ftypes):
		os.remove(dir_path +test_name +'csv.csv')

	return dumps, sizes, loads

class benchmark(object):
	def __init__(self, msg, fmt="%0.3g"):
		self.msg = msg
		self.fmt = fmt

	def __enter__(self):
		self.start = default_timer()
		return self

	def __exit__(self, *args):
		t = default_timer() - self.start
		print(("%s : " + self.fmt + " seconds") % (self.msg, t))
		self.time = t

if __name__ == '__main__':
	main(sys.argv[1:])
