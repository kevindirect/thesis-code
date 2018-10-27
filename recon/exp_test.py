# Kevin Patel

import sys
import os
from operator import itemgetter
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common_util import inner_join, remove_dups_list, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import dum
from recon.feat_util import gen_cluster_feats
from recon.label_util import gen_label_dfs, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct
from recon.split_util import get_train_test_split


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

		for lab_name, lab_df in gen_label_dfs(labels, labels_paths, asset):

			for lab_col_name in lab_df:
				logging.info(lab_col_name)
				lab_col_shf_df = lab_df[[lab_col_name]].dropna().shift(periods=-1, freq=None, axis=0).dropna().astype(int)
				prior_ser = lab_col_shf_df[lab_col_name].value_counts(normalize=True, sort=True)
				print("[priors]: 1: {:0.3f}, -1: {:0.3f}".format(prior_ser.loc[1], prior_ser.loc[-1]))

				for feat_df in gen_cluster_feats(features, features_paths, asset, km_info):
					feat_dum_df = pd.get_dummies(feat_df, prefix=feat_df.columns, prefix_sep='_', columns=feat_df.columns, drop_first=True)
					lab_feat_df = inner_join(lab_col_shf_df, feat_dum_df)

					true_vals, predicted = test_maxent(lab_feat_df.iloc[:, 1:], lab_feat_df.iloc[:, 0])
					print(", ".join(feat_df.columns) +": {:0.2f}".format(accuracy_score(true_vals, predicted)))

				print()


def test_maxent(feats, lab):
	feat_train, feat_test, lab_train, lab_test = get_train_test_split(feats, lab)
	logging.debug("feat_train.shape {0}, lab_train.shape {0}".format(feat_train.shape, lab_train.shape))
	logging.debug("feat_test.shape {0}, lab_test.shape {0}".format(feat_test.shape, lab_test.shape))
	assert(feat_train.shape[0] == lab_train.shape[0])
	assert(feat_test.shape[0]  == lab_test.shape[0])

	lr = LogisticRegression(penalty='l2', tol=0.001, C=0.5, fit_intercept=True, intercept_scaling=1, random_state=0)
	clf = lr.fit(feat_train, lab_train)
	predictions = clf.predict(feat_test)

	return lab_test, predictions


if __name__ == '__main__':
	test(sys.argv[1:])