
import numpy as np
import pandas as pd
import sys
import os
from os.path import dirname, realpath


ftypes = ['fixed_hdf', 'table_hdf', 'feather', 'parquet', 'csv']
get_script_dir = lambda: dirname(realpath(sys.argv[0])) +os.sep

dump_df = pd.read_csv(get_script_dir()+ 'dump_df.csv')
size_df = pd.read_csv(get_script_dir()+ 'size_df.csv')
load_df = pd.read_csv(get_script_dir()+ 'load_df.csv')

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
