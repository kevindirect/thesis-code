"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import exists, basename, isdir
from functools import partial
import pickle
import logging

import numpy as np
import pandas as pd

from common_util import JSON_SFX_LEN, NestedDefaultDict, load_json, load_df, isnt, is_type, compose, pd_rows_key_in, pd_add_2nd_level, pd_split_ternary_to_binary, midx_intersect, pd_get_midx_level, pd_rows, df_midx_restack, df_add_midx_level, in_debug_mode, benchmark
from model.common import XG_PROCESS_DIR, XG_DATA_DIR, XG_INDEX_FNAME, INTERVAL_YEARS
from recon.dataset_util import gen_group


""" ********** UTIL FNS ********** """
def xgload(xg_subset_dir):
	ndd = NestedDefaultDict()
	for d in os.listdir(xg_subset_dir):
		ddir = xg_subset_dir +d +sep
		if (isdir(ddir)):
			try:
				index = load_json(XG_INDEX_FNAME, dir_path=ddir)
			except FileNotFoundError as f:
				logging.warning(f'{d}{sep}{XG_INDEX_FNAME} not found, skipping...')
				continue
			else:
				for i, path in enumerate(index):
					ndd[path] = load_df(f'{i}.pickle', dir_path=ddir, data_format='pickle')
	return ndd

def get_xg_feature_dfs(asset, xg_f_dir=XG_DATA_DIR +'features' +sep, \
	overwrite_cache=False, blacklist=[]):
	"""
	Return reduced dictionary of feature dataframes for a given asset.
	XXX - hardcoded and inelegant
	"""
	cache_file = f"{xg_f_dir}{asset}.cache.pickle"

	if (overwrite_cache or not exists(cache_file)):
		f_store = xgload(xg_f_dir)
		flist = list(sorted(set([k[1] for k in f_store.childkeys([asset])])))
		faxe = {fq: [ft for ft in flist if (ft.startswith(fq) and ft not in blacklist)] \
			for fq in set([fq[0] for fq in flist])}

		# Load axefiles and paths into dicts
		axe_dict, path_dict = {}, {}
		for freq in faxe.keys():
			axe_dict[freq], path_dict[freq] = {}, {}
			for src in ['pba', 'vol', 'buzz', 'nonbuzz']:
				axe_dict[freq][src] = list(filter(partial(src_in_axefile, f_store=f_store, \
					asset=asset, src=src), faxe[freq]))
				path_dict[freq][src] = {}
				for axef in axe_dict[freq][src]:
					paths = f_store.childkeys([asset, axef])
					if (src in ('pba', 'vol')):
						path_dict[freq][src][axef] = list(filter(lambda path: path[-1] \
							.startswith(src), paths))
					elif (src in ('buzz', 'nonbuzz')):
						path_dict[freq][src][axef] = list(filter(lambda path: path[2] \
							.endswith('_'+src), paths))

		if (in_debug_mode()):
			for freq in path_dict.keys():
				pba_len = pd.Series(index=path_dict[freq]['pba'].keys(), \
					data=map(len, path_dict[freq]['pba'].values()), name='pba')
				vol_len = pd.Series(index=path_dict[freq]['vol'].keys(), \
					data=map(len, path_dict[freq]['vol'].values()), name='vol')
				buzz_len = pd.Series(index=path_dict[freq]['buzz'].keys(), \
					data=map(len, path_dict[freq]['buzz'].values()), name='buzz')
				nonbuzz_len = pd.Series(index=path_dict[freq]['nonbuzz'].keys(), \
					data=map(len, path_dict[freq]['nonbuzz'].values()), name='nonbuzz')
				comb_len = pd.concat([pba_len, vol_len, buzz_len, nonbuzz_len], \
					axis=1).fillna(0).astype(int)
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
						if (src in ('pba', 'vol') and \
							working_wrap[0][-1].split('_')[:2]==path[-1].split('_')[:2]):
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
							gr_df.index = gr_df.index.set_levels(['_'.join([sublvl, path[-1]]) \
								for sublvl in gr_df.index.levels[1]], level=1, inplace=False)
							to_subset.append(gr_df)
							subset_name_keys.extend(path[-1].split('_'))
							logging.debug(gr_df)

						# Disambiguate sub-dfs and set names
						subset_name = '_'.join(dict.fromkeys(subset_name_keys).keys())
						concat_df_dict[freq][src][axef][subset_name] = pd.concat(to_subset) \
							.sort_index(ascending=True) if (len(to_subset) > 1) else to_subset[0]
						logging.info(subset_name)

		with open(cache_file, 'wb') as f:
			pickle.dump(concat_df_dict, f)
	else:
		with open(cache_file, 'rb') as f:
			concat_df_dict = pickle.load(f)
	return concat_df_dict

def get_xg_label_target_dfs(asset, xg_l_dir=XG_DATA_DIR +'labels' +sep, xg_t_dir=XG_DATA_DIR +'targets' +sep, overwrite_cache=False):
	"""
	Return dictionaries of labels and targets for a given asset.
	XXX - this can be cleaned up - a lot of hardcoding and loops can be used for cleanup
	"""
	cache_files = (f"{xg_l_dir}{asset}.cache.pickle", f"{xg_t_dir}{asset}.cache.pickle")

	if (overwrite_cache or not all(map(exists, cache_files))):
		l_store, t_store = xgload(xg_l_dir), xgload(xg_t_dir)

		# ddir/dret
		ddir_pba_hoc = list(l_store.childkeys([asset, 'ddir', 'ddir', \
			'pba_hoc_hdxret_ddir']))
		ddir_vol_hoc = list(l_store.childkeys([asset, 'ddir', 'ddir', \
			'vol_hoc_hdxret_ddir']))
		dret_pba_hoc = list(t_store.childkeys([asset, 'dret', 'dret', \
			'pba_hoc_hdxret_dret']))
		dret_vol_hoc = list(t_store.childkeys([asset, 'dret', 'dret', \
			'vol_hoc_hdxret_dret']))

		# ddir1/dret1
		groups = ['lin', 'log']
		fmt3, fmt4 = '{}_{}', '{}_hdxret1_{}'
		e = 'ddir1'
		b = 'pba_hoc'; ddir1_pba_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; ddir1_pba_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; ddir1_vol_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; ddir1_vol_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		e = 'dret1'
		b = 'pba_hoc'; dret1_pba_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dret1_pba_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dret1_vol_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dret1_vol_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}

		# ddir2/dret2
		groups = ['avg', 'std', 'mad', 'max', 'min']
		fmt3, fmt4 = '{}_{}', '{}_hdxret2_{}'
		e = 'ddir2'
		b = 'pba_hoc'; ddir2_pba_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; ddir2_pba_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; ddir2_vol_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; ddir2_vol_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		e = 'dret2'
		b = 'pba_hoc'; dret2_pba_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dret2_pba_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dret2_vol_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dret2_vol_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}

		# dxfbdir1/dxbret1/dxfbval1
		groups = ['lin', 'log']
		fmt3, fmt4 = '{}_{}', '{}_hdxcret1_{}'
		e = 'dxfbdir1'
		b = 'pba_hoc'; dxfbdir1_pba_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dxfbdir1_pba_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dxfbdir1_vol_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dxfbdir1_vol_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		e = 'dxfbcret1'
		b = 'pba_hoc'; dxfbcret1_pba_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dxfbcret1_pba_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dxfbcret1_vol_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dxfbcret1_vol_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		e = 'dxfbval1'
		b = 'pba_hoc'; dxfbval1_pba_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dxfbval1_pba_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dxfbval1_vol_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dxfbval1_vol_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}

		# dxfbdir2/dxfbcret2/dxfbval2
		groups = ['avg', 'std', 'mad', 'max', 'min']
		fmt3, fmt4 = '{}_{}', '{}_hdxcret2_{}'
		e = 'dxfbdir2'
		b = 'pba_hoc'; dxfbdir2_pba_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dxfbdir2_pba_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dxfbdir2_vol_hoc = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dxfbdir2_vol_hlh = {g: list(l_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		e = 'dxfbcret2'
		b = 'pba_hoc'; dxfbcret2_pba_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dxfbcret2_pba_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dxfbcret2_vol_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dxfbcret2_vol_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		e = 'dxfbval2'
		b = 'pba_hoc'; dxfbval2_pba_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'pba_hlh'; dxfbval2_pba_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hoc'; dxfbval2_vol_hoc = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}
		b = 'vol_hlh'; dxfbval2_vol_hlh = {g: list(t_store.childkeys([asset, e, \
			fmt3.format(e, g), fmt4.format(b, e)])) for g in groups}

		l_df_dict = {
			'hoc': {
				'pba': {
					'ddir': get_lt_df(ddir_pba_hoc, l_store),
					'ddir1_log': get_lt_df(ddir1_pba_hoc['log'], l_store),
					'dxfbdir1_log': get_lt_df(dxfbdir1_pba_hoc['log'], l_store),
					'ddir2_avg': get_lt_df(ddir2_pba_hoc['avg'], l_store),
					'ddir2_std': get_lt_df(ddir2_pba_hoc['std'], l_store),
					'ddir2_mad': get_lt_df(ddir2_pba_hoc['mad'], l_store),
					'ddir2_max': get_lt_df(ddir2_pba_hoc['max'], l_store),
					'ddir2_min': get_lt_df(ddir2_pba_hoc['min'], l_store),
					'dxfbdir2_avg': get_lt_df(dxfbdir2_pba_hoc['avg'], l_store),
					'dxfbdir2_std': get_lt_df(dxfbdir2_pba_hoc['std'], l_store),
					'dxfbdir2_mad': get_lt_df(dxfbdir2_pba_hoc['mad'], l_store),
					'dxfbdir2_max': get_lt_df(dxfbdir2_pba_hoc['max'], l_store),
					'dxfbdir2_min': get_lt_df(dxfbdir2_pba_hoc['min'], l_store)
				},
				'vol': {
					'ddir': get_lt_df(ddir_vol_hoc, l_store),
					'ddir1_log': get_lt_df(ddir1_vol_hoc['log'], l_store),
					'dxfbdir1_log': get_lt_df(dxfbdir1_vol_hoc['log'], l_store),
					'ddir2_avg': get_lt_df(ddir2_vol_hoc['avg'], l_store),
					'ddir2_std': get_lt_df(ddir2_vol_hoc['std'], l_store),
					'ddir2_mad': get_lt_df(ddir2_vol_hoc['mad'], l_store),
					'ddir2_max': get_lt_df(ddir2_vol_hoc['max'], l_store),
					'ddir2_min': get_lt_df(ddir2_vol_hoc['min'], l_store),
					'dxfbdir2_avg': get_lt_df(dxfbdir2_vol_hoc['avg'], l_store),
					'dxfbdir2_std': get_lt_df(dxfbdir2_vol_hoc['std'], l_store),
					'dxfbdir2_mad': get_lt_df(dxfbdir2_vol_hoc['mad'], l_store),
					'dxfbdir2_max': get_lt_df(dxfbdir2_vol_hoc['max'], l_store),
					'dxfbdir2_min': get_lt_df(dxfbdir2_vol_hoc['min'], l_store)
				}
			}
		}
		t_df_dict = {
			'hoc': {
				'pba': {
					'dret': get_lt_df(dret_pba_hoc, t_store),
					'dret1_log': get_lt_df(dret1_pba_hoc['log'], t_store),
					'dxfbcret1_log': get_lt_df(dxfbcret1_pba_hoc['log'], t_store),
					'dxfbval1_log': get_lt_df(dxfbval1_pba_hoc['log'], t_store),
					'dret2_avg': get_lt_df(dret2_pba_hoc['avg'], t_store),
					'dret2_std': get_lt_df(dret2_pba_hoc['std'], t_store),
					'dret2_mad': get_lt_df(dret2_pba_hoc['mad'], t_store),
					'dret2_max': get_lt_df(dret2_pba_hoc['max'], t_store),
					'dret2_min': get_lt_df(dret2_pba_hoc['min'], t_store),
					'dxfbcret2_avg': get_lt_df(dxfbcret2_pba_hoc['avg'], t_store),
					'dxfbcret2_std': get_lt_df(dxfbcret2_pba_hoc['std'], t_store),
					'dxfbcret2_mad': get_lt_df(dxfbcret2_pba_hoc['mad'], t_store),
					'dxfbcret2_max': get_lt_df(dxfbcret2_pba_hoc['max'], t_store),
					'dxfbcret2_min': get_lt_df(dxfbcret2_pba_hoc['min'], t_store),
					'dxfbval2_avg': get_lt_df(dxfbval2_pba_hoc['avg'], t_store),
					'dxfbval2_std': get_lt_df(dxfbval2_pba_hoc['std'], t_store),
					'dxfbval2_mad': get_lt_df(dxfbval2_pba_hoc['mad'], t_store),
					'dxfbval2_max': get_lt_df(dxfbval2_pba_hoc['max'], t_store),
					'dxfbval2_min': get_lt_df(dxfbval2_pba_hoc['min'], t_store)
				},
				'vol': {
					'dret': get_lt_df(dret_vol_hoc, t_store),
					'dret1_log': get_lt_df(dret1_vol_hoc['log'], t_store),
					'dxfbcret1_log': get_lt_df(dxfbcret1_vol_hoc['log'], t_store),
					'dxfbval1_log': get_lt_df(dxfbval1_vol_hoc['log'], t_store),
					'dret2_avg': get_lt_df(dret2_vol_hoc['avg'], t_store),
					'dret2_std': get_lt_df(dret2_vol_hoc['std'], t_store),
					'dret2_mad': get_lt_df(dret2_vol_hoc['mad'], t_store),
					'dret2_max': get_lt_df(dret2_vol_hoc['max'], t_store),
					'dret2_min': get_lt_df(dret2_vol_hoc['min'], t_store),
					'dxfbcret2_avg': get_lt_df(dxfbcret2_vol_hoc['avg'], t_store),
					'dxfbcret2_std': get_lt_df(dxfbcret2_vol_hoc['std'], t_store),
					'dxfbcret2_mad': get_lt_df(dxfbcret2_vol_hoc['mad'], t_store),
					'dxfbcret2_max': get_lt_df(dxfbcret2_vol_hoc['max'], t_store),
					'dxfbcret2_min': get_lt_df(dxfbcret2_vol_hoc['min'], t_store),
					'dxfbval2_avg': get_lt_df(dxfbval2_vol_hoc['avg'], t_store),
					'dxfbval2_std': get_lt_df(dxfbval2_vol_hoc['std'], t_store),
					'dxfbval2_mad': get_lt_df(dxfbval2_vol_hoc['mad'], t_store),
					'dxfbval2_max': get_lt_df(dxfbval2_vol_hoc['max'], t_store),
					'dxfbval2_min': get_lt_df(dxfbval2_vol_hoc['min'], t_store)
				}
			}
		}
		with open(cache_files[0], 'wb') as f:
			pickle.dump(l_df_dict, f)
		with open(cache_files[1], 'wb') as f:
			pickle.dump(t_df_dict, f)
	else:
		with open(cache_files[0], 'rb') as f:
			l_df_dict = pickle.load(f)
		with open(cache_files[1], 'rb') as f:
			t_df_dict = pickle.load(f)
	return l_df_dict, t_df_dict


""" ********** HARDCODED DATA GETTER FNS ********** """
def get_hardcoded_daily_feature_dfs(fd, src, cat=True):
	"""
	Return hardcoded daily freq feature dfs to use in experiments.
	"""
	feature_dfs = []

	if (src in ('pba', 'vol')):
		feature_dfs.append(fd['d'][src]['dlogret']['{src}_dlh_dlogret'.format(src=src)])

	if (src in ('pba', 'vol', 'buzz')):
		for axe in ('dffd', 'dwrmx', 'dwrod', 'dwrpt', 'dwrzn'):
			for fdf in fd['d'][src][axe].values():
				# feature_dfs.append(df_filter_by_keywords(fdf, \
				# ('avgPrice', 'open', 'high', 'low', 'close')))
				feature_dfs.append(fdf)

	if (src in ('nonbuzz',)):
		for axe in ('dffd', 'dwrxmx'):
			for fdf in fd['d'][src][axe].values():
				feature_dfs.append(fdf)

	# return pd.concat(feature_dfs).sort_index(axis=0)
	return pd.concat(feature_dfs) if (cat) else feature_dfs

def get_hardcoded_hourly_feature_dfs(fd, src, cat=True, ret='logret'):
	"""
	Return hardcoded hourly freq feature dfs to use in experiments.
	"""
	def pd_modify_midx(pd_obj, sfx=None, offset=0):
		_sfx = '' if (isnt(sfx)) else f'_{sfx}'
		modified = ['_'.join([str(i+offset), ser_name.split('_')[1]]) +_sfx\
			for i, ser_name in enumerate(pd_obj.index.levels[1])]
		pd_obj.index = pd_obj.index.set_levels(modified, level=1)
		return pd_obj

	feature_dfs = []

	if (src in ('pba', 'vol')):
		ret_type = f'h{ret}'
		axe = bar = 'hohlca'
		off1 = 0

		sub_dfs = [pd_modify_midx(fd['h'][src][axe][f'{src}_{axe}'])]
		off2 = len(sub_dfs[-1].index.levels[1])
		for key, fdf in sorted(fd['h'][src][ret_type].items()):
			sub_dfs.append(pd_modify_midx(fdf, sfx=ret, offset=off2))
			off2 += len(sub_dfs[-1].index.levels[1])
		feature_dfs.append(pd_add_2nd_level(pd.concat(sub_dfs, axis=0), \
			keys=[f'{off1}_{src}_{axe}']))
		off1 += 1

		for axe in ('hdpt', 'hdod', 'hdmx', 'hdzn'):
			sub_dfs = [pd_modify_midx(fd['h'][src][axe][f'{src}_{bar}_{axe}'])]
			off2 = len(sub_dfs[-1].index.levels[1])
			for key, fdf in sorted(fd['h'][src][axe].items()):
				if (ret_type in key):
					midx_keys = tuple(key for key in fdf.index.levels[1] \
						if (ret_type in key))
					sub_dfs.append(pd_modify_midx(pd_rows_key_in(fdf, 'id1', midx_keys), \
						sfx=ret, offset=off2))
					off2 += len(sub_dfs[-1].index.levels[1])
			feature_dfs.append(pd_add_2nd_level(pd.concat(sub_dfs, axis=0), \
				keys=[f'{off1}_{src}_{axe}']))
			off1 += 1

		for axel in ('hdmx_hduni', 'hdzn_hdgau'):
			axe = axel.split('_')[1]
			sub_dfs = [pd_modify_midx(fd['h'][src][axe][f'{src}_{bar}_{axel}(8)'])]
			off2 = len(sub_dfs[-1].index.levels[1])
			for key, fdf in sorted(fd['h'][src][axe].items()):
				if (ret_type in key):
					midx_keys = tuple(key for key in fdf.index.levels[1] \
						if (ret_type in key))
					sub_dfs.append(pd_modify_midx(pd_rows_key_in(fdf, 'id1', midx_keys), \
						sfx=ret, offset=off2))
					off2 += len(sub_dfs[-1].index.levels[1])
			feature_dfs.append(pd_add_2nd_level(pd.concat(sub_dfs, axis=0), \
				keys=[f'{off1}_{src}_{axe}']))
			off1 += 1

	elif (src in ('buzz',)):
		for fdf in fd['h'][src][axe].values():
			feature_dfs.append(df_filter_by_keywords(fdf, ('open',)))

	return pd.concat(feature_dfs, axis=0) if (cat) else feature_dfs

def get_hardcoded_feature_dfs(fd, src, cat=True):
	"""
	Multiplex between different feature frequencies and return data.
	"""
	src_k = src.split('_')
	return {
		'd': get_hardcoded_daily_feature_dfs,
		'h': get_hardcoded_hourly_feature_dfs
	}.get(src_k[0])(fd, src_k[1], cat)

def get_hardcoded_label_target_dfs(ld, td, src):
	"""
	Return hardcoded label/target dfs to use in experiments.
	"""
	if (src == 'ddir'):
		# predict daily direction: ddir(t)
		ldf = pd_split_ternary_to_binary(ld['hoc']['pba']['ddir'] \
			.replace(to_replace=-1, value=0))
		tdf = pd_split_ternary_to_binary(td['hoc']['pba']['dret'])
	elif (src == 'ddir1'):
		# predict thresholded direction: ddir1(t)
		ldf_len = len(ld['hoc']['pba']['ddir1_log'].columns)
		ldf = pd_split_ternary_to_binary(ld['hoc']['pba']['ddir1_log'])
		ldf = ldf.sum(level=0) + len(ldf.columns)
		ldf = (ldf[ldf!=0] + ldf_len).fillna(0).astype(int)
		ldf = df_add_midx_level(ldf, 'pba_hoc_hdxret1_ddir1', loc=1, name='id1')

		tdf = pd_split_ternary_to_binary(td['hoc']['pba']['dret1_log']) \
			.abs().max(level=0)
		tdf = df_add_midx_level(tdf, 'pba_hoc_hdxret1_dret1', loc=1, name='id1')
		# tdf = tdf.sum(level=0) + len(tdf.columns)
		# tdf
	elif (src == 'dxfbdir1'):
		raise NotImplementedError()
	elif (src == 'ddir2'):
		raise NotImplementedError()
	elif (src == 'dxfbdir2'):
		raise NotImplementedError()

	return ldf, tdf


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

def df_filter_by_keywords(fdf, keywords):
	cols = [col for col in fdf.index.levels[1] if (any(keyword in col for keyword in keywords))]
	return pd_rows_key_in(fdf, 'id1', cols)

def get_common_interval_data(fdata, ldata, tdata, interval=INTERVAL_YEARS):
	"""
	Intersect common data over interval and return it
	"""
	com_idx = midx_intersect(pd_get_midx_level(fdata), pd_get_midx_level(ldata), \
		pd_get_midx_level(tdata))
	com_idx = com_idx[(com_idx > str(interval[0])) & (com_idx < str(interval[1]))]
	feature_df, label_df, target_df = map(compose(partial(pd_rows, idx=com_idx), \
		df_midx_restack), [fdata, ldata, tdata])
	assert(all(feature_df.index.levels[0]==label_df.index.levels[0]))
	assert(all(feature_df.index.levels[0]==target_df.index.levels[0]))
	return feature_df, label_df, target_df

