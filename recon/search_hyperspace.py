# Kevin Patel

import sys
import os
import getopt
from operator import itemgetter
import logging

import numpy as np
import pandas as pd
from dask_ml.model_selection import GridSearchCV

from common_util import RECON_DIR, load_json, inner_join, remove_dups_list, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import default_pipefile, default_cv_file
from recon.pipe_util import extract_pipeline
from recon.cv_util import extract_cv_splitter
from recon.feat_util import gen_split_feats
from recon.label_util import gen_label_dfs, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct
# from recon.hyperfit import cv_hyper_fit


def fit_all(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	usage = lambda: print('fit_all.py [-p <pipefile> -c <cv_file> -a <asset>]')
	pipefile = default_pipefile
	cv_file = default_cv_file
	relevant_assets = None

	try:
		opts, args = getopt.getopt(argv, 'hp:c:a:', ['help', 'pipefile=', 'cv_file=', 'asset='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-p', '--pipefile'):
			pipefile = arg
		elif opt in ('-c', '--cv_file'):
			cv_file = arg
		elif opt in ('-a', '--asset'):
			relevant_assets = [arg]

	pipe_dict = load_json(pipefile, dir_path=RECON_DIR)
	pipeline, grid = extract_pipeline(pipe_dict)
	logging.info('loaded pipeline from ' +str(pipefile))

	cv_dict = load_json(cv_file, dir_path=RECON_DIR)
	cv_splitter = extract_cv_splitter(cv_dict)
	logging.info('loaded cross val settings from ' +str(cv_file))

	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax'], cs2['sax']['dzn_sax'], subset=['raw_pba', 'raw_vol'])
	logging.info('loaded features')

	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb'], cs2['labels']['itb'])
	logging.info('loaded labels')

	valid_assets = remove_dups_list(map(itemgetter(0), features_paths))
	relevant_assets = valid_assets if (relevant_assets is None) else relevant_assets
	assert(set(relevant_assets) <= set(valid_assets))
	logging.info('running on the following asset(s): ' +', '.join(relevant_assets))


	for asset in relevant_assets:
		logging.info('asset: ' +str(asset))

		for lab_df in gen_label_dfs(labels, labels_paths, asset):

			for lab_col_name in lab_df:
				logging.info(lab_col_name)
				lab_col_shf_df = lab_df[[lab_col_name]].dropna().shift(periods=-1, freq=None, axis=0).dropna().astype(int)
				prior_ser = lab_col_shf_df[lab_col_name].value_counts(normalize=True, sort=True)
				print("[priors]: 1: {:0.3f}, -1: {:0.3f}".format(prior_ser.loc[1], prior_ser.loc[-1]))

				for one_feat_df in gen_split_feats(features, features_paths, asset):
					lab_feat_df = inner_join(lab_col_shf_df, one_feat_df)
					print(lab_feat_df)
				break
			break
		break
		

if __name__ == '__main__':
	fit_all(sys.argv[1:])