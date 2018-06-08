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
from recon.feat_util import split_ser
from recon.label_util import get_base_labels

def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax'], cs2['sax']['dzn_sax'], subset=['raw_pba', 'raw_vol'])
	logging.info('loaded features')

	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb'], cs2['labels']['itb'])
	logging.info('loaded labels')

	# print('features')
	# for feature_path in features_paths:
	# 	print('_'.join(feature_path))
	# 	fdict = list_get_dict(features, feature_path)
	# 	print(fdict.columns)

	# print('labels')
	# for label_path in labels_paths:
	# 	print('_'.join(label_path))
	# 	ldict = list_get_dict(labels, label_path)
	# 	print(get_base_labels(ldict))
	# 	print(ldict.iloc[:, 0].value_counts())
	# 	print(ldict.iloc[:, 0].head())

	assets = remove_dups_list(map(itemgetter(0), features_paths))

	for asset in assets:
		logging.info('asset: ' +str(asset))

		for feature_path in filter(lambda fpath: fpath[0]==asset, features_paths):
			feat_df = list_get_dict(features, feature_path)

			for col_name in feat_df:
				sax_df = split_ser(feat_df[col_name], 8, pfx='_'.join(feature_path[1:]))
				print(sax_df)
				print(count_nn_df(sax_df))
				kmeans = KMeans(n_clusters=4, random_state=0).fit(sax_df.values)
				print(kmeans.labels_)
				break


	
if __name__ == '__main__':
	test(sys.argv[1:])