# Kevin Patel

import sys
import os
from operator import itemgetter
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common_util import search_df, get_subset, count_nn_df, remove_dups_list, list_get_dict, list_set_dict, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import dum
from recon.feat_util import split_ser, handle_nans_df
from recon.label_util import get_base_labels, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct

def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax'], cs2['sax']['dzn_sax'], subset=['raw_pba', 'raw_vol'])
	logging.info('loaded features')

	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb'], cs2['labels']['itb'])
	logging.info('loaded labels')

	assets = remove_dups_list(map(itemgetter(0), features_paths))
	logging.info('all assets: ' +', '.join(assets))

	for asset in assets:
		logging.info('asset: ' +str(asset))

		for label_path in filter(lambda lpath: lpath[0]==asset, labels_paths):
			label_path_str = '_'.join(label_path[1:])
			logging.info('label df: ' +label_path_str)
			lab_df = list_get_dict(labels, label_path)
			lab_fct_df = pd.DataFrame(index=lab_df.index)

			# lab_col_set_selectors = {base_label: get_subset(lab_df.columns, make_sw_dict(base_label))
			# 	for base_label in get_base_labels(lab_df.columns)}

			# Iterate through all variations of this label
			for base_label in get_base_labels(lab_df.columns):
				logging.debug('base label: ' +base_label)
				# lab_name_prefix = '_'.join([label_path_str, base_label])
				base_label_subset = get_subset(lab_df.columns, make_sw_dict(base_label))
				dir_col_name = '_'.join([base_label, 'dir'])
				dir_col = default_fct(lab_df[base_label_subset], name_pfx=base_label)[dir_col_name]
				lab_fct_df[dir_col_name] = dir_col

			lab_fct_df.index = lab_fct_df.index.normalize()
			print(lab_fct_df)
			return

			# Iterate through all feature sets
			for feature_path in filter(lambda fpath: fpath[0]==asset, features_paths):
				feat_df = list_get_dict(features, feature_path)
				np_feat = {}

				for col_name in feat_df:
					col_name_prefix = '_'.join(feature_path[1:] +[col_name])
					logging.info(col_name_prefix)
					sax_df = handle_nans_df(split_ser(feat_df[col_name], 8, pfx=col_name_prefix))
					# print('num_rows:', count_nn_df(sax_df).iloc[0])

					kmeans = KMeans(n_clusters=4, random_state=0).fit(sax_df.values)
					np_feat[col_name_prefix +'_' +'kmeans(4)'] = kmeans.labels_

				print(np_feat)


def make_sw_dict(sw_str):
	return {
		"exact": [],
		"startswith": [],
		"endswith": [sw_str],
		"regex": [],
		"exclude": None
	}

def make_ew_dict(ew_str):
	return {
		"exact": [],
		"startswith": [],
		"endswith": [ew_str],
		"regex": [],
		"exclude": None
	}


if __name__ == '__main__':
	test(sys.argv[1:])