"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import exists, basename
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import JSON_SFX_LEN, NestedDefaultDict, load_json, load_df, is_type, in_debug_mode, benchmark
from model.common import XG_PROCESS_DIR, XG_DATA_DIR, XG_INDEX_FNAME
from recon.dataset_util import gen_group


""" ********** UTIL FNS ********** """
def xgload(xg_subset_dir):
	ndd = NestedDefaultDict()
	for d in os.listdir(xg_subset_dir):
		ddir = xg_subset_dir +d +sep
		try:
			index = load_json(XG_INDEX_FNAME, dir_path=ddir)
		except FileNotFoundError as f:
			logging.warning('{}{}{} not found, skipping...'.format(d, sep, XG_INDEX_FNAME))
			continue
		else:
			for i, path in enumerate(index):
				ndd[path] = load_df('{}.pickle'.format(i), dir_path=ddir, data_format='pickle')
	return ndd

def get_xg_feature_dfs(asset, xg_f_dir=XG_DATA_DIR +'features' +sep, blacklist=[]):
	"""
	Return reduced dictionary of feature dataframes for a given asset.
	XXX - hardcoded and inelegant
	"""
	f_store = xgload(xg_f_dir)
	flist = list(sorted(set([k[1] for k in f_store.childkeys([asset])])))
	faxe = {fq: [ft for ft in flist if (ft.startswith(fq) and ft not in blacklist)] for fq in set([fq[0] for fq in flist])}

	# Load axefiles and paths into dicts
	axe_dict, path_dict = {}, {}
	for freq in faxe.keys():
		axe_dict[freq], path_dict[freq] = {}, {}
		for src in ['pba', 'vol', 'buzz', 'nonbuzz']:
			axe_dict[freq][src] = list(filter(partial(src_in_axefile, f_store=f_store, asset=asset, src=src), faxe[freq]))
			path_dict[freq][src] = {}
			for axef in axe_dict[freq][src]:
				paths = f_store.childkeys([asset, axef])
				if (src in ('pba', 'vol')):
					path_dict[freq][src][axef] = list(filter(lambda path: path[-1].startswith(src), paths))
				elif (src in ('buzz', 'nonbuzz')):
					path_dict[freq][src][axef] = list(filter(lambda path: path[2].endswith('_'+src), paths))

	if (in_debug_mode()):
		for freq in path_dict.keys():
			pba_len = pd.Series(index=path_dict[freq]['pba'].keys(),data=map(len, path_dict[freq]['pba'].values()), name='pba')
			vol_len = pd.Series(index=path_dict[freq]['vol'].keys(),data=map(len, path_dict[freq]['vol'].values()), name='vol')
			buzz_len = pd.Series(index=path_dict[freq]['buzz'].keys(),data=map(len, path_dict[freq]['buzz'].values()), name='buzz')
			nonbuzz_len = pd.Series(index=path_dict[freq]['nonbuzz'].keys(),data=map(len, path_dict[freq]['nonbuzz'].values()), name='nonbuzz')
			comb_len = pd.concat([pba_len, vol_len, buzz_len, nonbuzz_len], axis=1).fillna(0).astype(int)
			assert(all(comb_len.loc[:, 'pba']==comb_len.loc[:, 'vol']))

	# Wrap related paths in outer list to prepare to concatenate their dfs into single df
	concat_path_dict = {}
	for freq in path_dict.keys():
		concat_path_dict[freq] = {}
		for src in path_dict[freq].keys():
			concat_path_dict[freq][src] = {}
			for axef, paths in path_dict[freq][src].items():
				new_paths = paths
				if (axef in ('hdgau', 'hduni')):
					new_paths = [path for path in new_paths if ('8' in path[-1])] # Filter out all symbol sizes except 8 from hdgau and hduni
				if (len(new_paths)==0):
					continue
				working_wrap, all_wraps = [new_paths[0]], []
				for path in new_paths[1:]:
					if (src in ('pba', 'vol') and working_wrap[0][-1].split('_')[:2]==path[-1].split('_')[:2]):
						working_wrap.append(path)
					elif (src in ('buzz', 'nonbuzz') and working_wrap[0][2]==path[2]):
						working_wrap.append(path)
					else:
						all_wraps.append(working_wrap)
						working_wrap = [path]
				all_wraps.append(working_wrap)
				concat_path_dict[freq][src][axef] = all_wraps

	# Concatenate related dfs to reduce final number of dataframes
	concat_df_dict = {}
	for freq in concat_path_dict.keys():
		concat_df_dict[freq] = {}
		for src in concat_path_dict[freq].keys():
			concat_df_dict[freq][src] = {}
			for axef, path_groups in concat_path_dict[freq][src].items():
				concat_df_dict[freq][src][axef] = {}
				for g in path_groups:
					logging.debug([path[-1] for path in g])
					to_subset, subset_name_keys = [], []
					for path in g:
						gr_df = f_store[path]
						logging.debug(gr_df.index.levels[1])
						# Append last path key to 2nd (last) level of MultiIndex
						gr_df.index = gr_df.index.set_levels(['_'.join([sublvl, path[-1]]) for sublvl in gr_df.index.levels[1]], level=1, inplace=False)
						to_subset.append(gr_df)
						subset_name_keys.extend(path[-1].split('_'))
						logging.debug(gr_df)

					# Disambiguate sub-dfs and set names
					subset_name = '_'.join(dict.fromkeys(subset_name_keys).keys())
					concat_df_dict[freq][src][axef][subset_name] = pd.concat(to_subset).sort_index(ascending=True) if (len(to_subset) > 1) else to_subset[0]
					logging.info(subset_name)
	return concat_df_dict

def get_xg_label_target_dfs(asset, xg_l_dir=XG_DATA_DIR +'labels' +sep, xg_t_dir=XG_DATA_DIR +'targets' +sep):
	"""
	Return dictionaries of labels and targets for a given asset.
	XXX - this can be cleaned up - a lot of hardcoding and loops can be used for cleanup
	"""
	l_store, t_store = xgload(xg_l_dir), xgload(xg_t_dir)

	# ddir/dret
	ddir_pba_hoc = list(l_store.childkeys([asset, 'ddir', 'ddir', 'pba_hoc_hdxret_ddir']))
	ddir_vol_hoc = list(l_store.childkeys([asset, 'ddir', 'ddir', 'vol_hoc_hdxret_ddir']))
	dret_pba_hoc = list(t_store.childkeys([asset, 'dret', 'dret', 'pba_hoc_hdxret_dret']))
	dret_vol_hoc = list(t_store.childkeys([asset, 'dret', 'dret', 'vol_hoc_hdxret_dret']))

	# ddir1/dret1
	groups = ['lin', 'log']
	fmt3, fmt4 = '{}_{}', '{}_hdxret1_{}'
	e = 'ddir1'
	b = 'pba_hoc'; ddir1_pba_hoc = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'pba_hlh'; ddir1_pba_hlh = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hoc'; ddir1_vol_hoc = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hlh'; ddir1_vol_hlh = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	e = 'dret1'
	b = 'pba_hoc'; dret1_pba_hoc = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'pba_hlh'; dret1_pba_hlh = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hoc'; dret1_vol_hoc = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hlh'; dret1_vol_hlh = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}

	# ddir2/dret2
	scalars = ['0.5', '1', '2']
	stats = ['avg', 'std', 'mad', 'max', 'min']
	fmt4, fmt5 = '{}_hdxret2_{}', '{}_hdxret2({}*{},1)_{}'
	e = 'ddir2'
	b = 'pba_hoc'; ddir2_pba_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'pba_hlh'; ddir2_pba_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hoc'; ddir2_vol_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hlh'; ddir2_vol_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	e = 'dret2'
	b = 'pba_hoc'; dret2_pba_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'pba_hlh'; dret2_pba_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hoc'; dret2_vol_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hlh'; dret2_vol_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}

	# dxfbdir1/dxbret1/dxfbval1
	groups = ['lin', 'log']
	fmt3, fmt4 = '{}_{}', '{}_hdxcret1_{}'
	e = 'dxfbdir1'
	b = 'pba_hoc'; dxfbdir1_pba_hoc = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'pba_hlh'; dxfbdir1_pba_hlh = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hoc'; dxfbdir1_vol_hoc = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hlh'; dxfbdir1_vol_hlh = {g: list(l_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	e = 'dxfbcret1'
	b = 'pba_hoc'; dxfbcret1_pba_hoc = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'pba_hlh'; dxfbcret1_pba_hlh = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hoc'; dxfbcret1_vol_hoc = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hlh'; dxfbcret1_vol_hlh = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	e = 'dxfbval1'
	b = 'pba_hoc'; dxfbval1_pba_hoc = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'pba_hlh'; dxfbval1_pba_hlh = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hoc'; dxfbval1_vol_hoc = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
	b = 'vol_hlh'; dxfbval1_vol_hlh = {g: list(t_store.childkeys([asset, e, fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}

	# dxfbdir2/dxbret2/dxfbval2
	scalars = ['0.5', '1', '2']
	stats = ['avg', 'std', 'mad', 'max', 'min']
	fmt4, fmt5 = '{}_hdxcret2_{}', '{}_hdxcret2({}*{},1)_{}'
	e = 'dxfbdir2'
	b = 'pba_hoc'; dxfbdir2_pba_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'pba_hlh'; dxfbdir2_pba_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hoc'; dxfbdir2_vol_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hlh'; dxfbdir2_vol_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	e = 'dxfbcret2'
	b = 'pba_hoc'; dxfbcret2_pba_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'pba_hlh'; dxfbcret2_pba_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hoc'; dxfbcret2_vol_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hlh'; dxfbcret2_vol_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	e = 'dxfbval2'
	b = 'pba_hoc'; dxfbval2_pba_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'pba_hlh'; dxfbval2_pba_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hoc'; dxfbval2_vol_hoc = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}
	b = 'vol_hlh'; dxfbval2_vol_hlh = {d: [[asset, e, e, fmt4.format(b, e), fmt5.format(b, c, d, e)] for c in scalars] for d in stats}

	return ({
		'hoc': {
			'pba': {
				'ddir': get_lt_df(ddir_pba_hoc, l_store),
				'ddir1': get_lt_df(ddir1_pba_hoc['log'], l_store),
				'dxfbdir1': get_lt_df(dxfbdir1_pba_hoc['log'], l_store)
			},
			'vol': {
				'ddir': get_lt_df(ddir_vol_hoc, l_store),
				'ddir1': get_lt_df(ddir1_vol_hoc['log'], l_store),
				'dxfbdir1': get_lt_df(dxfbdir1_vol_hoc['log'], l_store)
			}
		}
	}, {
		'hoc': {
			'pba': {
				'dret': get_lt_df(dret_pba_hoc, t_store),
				'dret1': get_lt_df(dret1_pba_hoc['log'], t_store),
				'dxfbcret1': get_lt_df(dxfbcret1_pba_hoc['log'], t_store),
				'dxfbval1': get_lt_df(dxfbval1_pba_hoc['log'], t_store)
			},
			'vol': {
				'dret': get_lt_df(dret_vol_hoc, t_store),
				'dret1': get_lt_df(dret1_vol_hoc['log'], t_store),
				'dxfbcret1': get_lt_df(dxfbcret1_vol_hoc['log'], t_store),
				'dxfbval1': get_lt_df(dxfbval1_vol_hoc['log'], t_store)
			}
		}
	})


""" ********** HELPER FNS ********** """
def src_in_axefile(axefile, f_store, asset, src):
	"""
	Return True if the supplied axefile's paths contains the src string anywhere for the given asset.
	"""
	axefile = [axefile] if (is_type(axefile, str)) else axefile
	return any(any(src in val for val in path) for path in f_store.childkeys([asset, *axefile]))

def get_lt_df(paths, lt_store):
	"""
	Return label or target data as concatenated DataFrame from list of paths and lt_store.
	"""
	lt_df = pd.concat([lt_store[path] for path in paths], axis=1, keys=[path[-1] for path in paths])
	lt_df.columns = lt_df.columns.droplevel(-1)
	return lt_df

