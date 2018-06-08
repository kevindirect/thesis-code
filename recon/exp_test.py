# Kevin Patel

import sys
import os
from operator import itemgetter
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common_util import search_df, get_subset, list_get_dict, list_set_dict, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import dum
from recon.label_util import get_base_labels

def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	# Load feature and label data
	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax'], cs2['sax']['dzn_sax'], subset=['raw_pba', 'raw_vol'])
	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb'], cs2['labels']['itb'])

	# Asset list
	assets = list(set(map(itemgetter(0), features_paths)))
	print(assets)

	print('features')
	for feature_path in features_paths:
		print('_'.join(feature_path))
		fdict = list_get_dict(features, feature_path)
		print(fdict.columns)

	print('labels')
	for label_path in labels_paths:
		print('_'.join(label_path))
		ldict = list_get_dict(labels, label_path)
		print(get_base_labels(ldict))
		print(ldict.iloc[:, 0].value_counts())
		print(ldict.iloc[:, 0].head())

	# kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
	
if __name__ == '__main__':
	test(sys.argv[1:])