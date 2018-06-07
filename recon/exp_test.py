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
from data.access_util import df_getters as dg
from recon.common import dum

def split_sax(ser):
	# split_df = pd.DataFrame(index=ser.index)
	return ser.str.split(',', 1, expand=True)

def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax'])
	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb'])
	assets = list(set(map(itemgetter(0), features_paths)))
	print(assets)

	sax_df = list_get_dict(features, features_paths[0])
	print(split_sax(sax_df['pba_avgPrice']))

	print('features')
	for feature_path in features_paths:
		print('_'.join(feature_path))

	print('labels')
	for label_path in labels_paths:
		print('_'.join(label_path))
	
	kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

	
if __name__ == '__main__':
	test(sys.argv[1:])