# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


from common_util import search_df, get_subset, dict_path, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg
from recon.common import dum


def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax'])
	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb']['itb_fth_of_xwhole'])

	print('features')
	print(features_paths)

	print('labels')
	print(labels_paths)
	
	kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

	
if __name__ == '__main__':
	test(sys.argv[1:])