# Kevin Patel

import sys
import os
from operator import itemgetter
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from common_util import get_custom_biz_freq, get_subset, inner_join, count_nn_df, remove_dups_list, list_get_dict, list_set_dict, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import dum
from recon.feat_util import gen_cluster_feats
from recon.label_util import gen_label_dfs, get_base_labels, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct

def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	km_info = {'cl': KMeans(n_clusters=8, random_state=0), 'sfx': 'kmeans(8)'}
	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax'], cs2['sax']['dzn_sax'], subset=['raw_pba', 'raw_vol'])
	logging.info('loaded features')

	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb'], cs2['labels']['itb'])
	logging.info('loaded labels')

	assets = remove_dups_list(map(itemgetter(0), features_paths))
	logging.info('all assets: ' +', '.join(assets))

	for asset in assets:
		logging.info('asset: ' +str(asset))

		for lab_df in gen_label_dfs(labels, labels_paths, asset):
			print(lab_df.columns)

		for feat_df in gen_cluster_feats(features, features_paths, asset, km_info):
			print(feat_df.columns)


			# for label_fct_col in label_fct_df:
			# 	label_fct_shf_df = label_fct_df[[label_fct_col]].dropna()
			# 	shift_freq = get_custom_biz_freq(label_fct_shf_df)
			# 	label_fct_shf_df = label_fct_shf_df.shift(periods=-1, freq=None, axis=0).dropna()

			# 	# Iterate through all feature sets
			# 	for feature_path in filter(lambda fpath: fpath[0]==asset, features_paths):
			# 		feat_df = list_get_dict(features, feature_path)
			# 		np_feat = {}
			# 		handled_df = pd.DataFrame(index=feat_df.index)
			# 		handled_df[label_fct_col] = label_fct_shf_df[label_fct_col]

			# 		for col_name in feat_df:
			# 			col_name_prefix = '_'.join(feature_path[1:] +[col_name])
			# 			logging.info(col_name_prefix)
			# 			sax_df = handle_nans_df(split_ser(feat_df[col_name], 8, pfx=col_name_prefix))
			# 			temp_df = inner_join(label_fct_shf_df, sax_df)
			# 			feats_only = temp_df[temp_df.columns[1:]]
			# 			# handled_df[sax_df.columns] = sax_df
			# 			# print('num_rows:', count_nn_df(sax_df).iloc[0])

			# 			kmeans = KMeans(n_clusters=4, random_state=0).fit(feats_only.values)
			# 			handled_df[col_name_prefix +'_' +'kmeans(4)'] = kmeans.labels_
			# 			pd.DataFrame(data=labels, columns=['cluster'], index=collapsed.index)
			# 	print(handled_df)

			# 	print(np_feat)


def test_logistic_reg(lab_feat_df):
	pass

# def make_sw_dict(sw_str):
# 	return {
# 		"exact": [],
# 		"startswith": [sw_str],
# 		"endswith": [],
# 		"regex": [],
# 		"exclude": None
# 	}

# def make_ew_dict(ew_str):
# 	return {
# 		"exact": [],
# 		"startswith": [],
# 		"endswith": [ew_str],
# 		"regex": [],
# 		"exclude": None
# 	}


if __name__ == '__main__':
	test(sys.argv[1:])