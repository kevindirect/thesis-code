# Kevin Patel

import sys
import os
import getopt
from operator import itemgetter
import logging

import numpy as np
import pandas as pd

from common_util import RECON_DIR, load_json, dump_df, inner_join, remove_dups_list, benchmark
from data.data_api import DataAPI
from data.access_util import df_getters as dg, col_subsetters2 as cs2
from recon.common import default_pipefile, default_cv_file
from recon.pipe_util import extract_pipeline
from recon.cv_util import extract_cv_splitter
from recon.feat_util import gen_split_feats
from recon.label_util import gen_label_dfs, default_fct, fastbreak_fct, confidence_fct, fastbreak_confidence_fct


def search_hyperspace(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	usage = lambda: print('search_hyperspace.py [-p <pipefile> -c <cv_file> -a <asset>]')
	optimize = False
	pipefile = default_pipefile
	cv_file = default_cv_file
	relevant_assets = None

	try:
		opts, args = getopt.getopt(argv, 'hOp:c:a:', ['help', 'optimize', 'pipefile=', 'cv_file=', 'asset='])
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ('-O', '--optimize'):
			optimize = True
		elif opt in ('-p', '--pipefile'):
			pipefile = arg
		elif opt in ('-c', '--cv_file'):
			cv_file = arg
		elif opt in ('-a', '--asset'):
			relevant_assets = [arg]

	if (optimize):
		logging.info('optimize flag set (parallelism enabled)')
		from dask_ml.model_selection import GridSearchCV
		# from multiprocessing import Pool
		# pool = Pool(processes=os.cpu_count())
	else:
		logging.info('optimize flag not set (parallelism disabled)')
		from sklearn.model_selection import GridSearchCV

	pipe_dict = load_json(pipefile, dir_path=RECON_DIR)
	pipeline, grid = extract_pipeline(pipe_dict)
	logging.info('loaded pipeline settings from ' +str(pipefile))

	cv_dict = load_json(cv_file, dir_path=RECON_DIR)
	cv_splitter = extract_cv_splitter(cv_dict)
	logging.info('loaded cross val settings from ' +str(cv_file))

	features_paths, features = DataAPI.load_from_dg(dg['sax']['dzn_sax_test'], cs2['sax']['dzn_sax_test'], subset=['raw_pba', 'raw_vol'])
	logging.info('loaded features')

	labels_paths, labels = DataAPI.load_from_dg(dg['labels']['itb_test'], cs2['labels']['itb_test'])
	print(labels)
	logging.info('loaded labels')

	valid_assets = remove_dups_list(map(itemgetter(0), features_paths))
	relevant_assets = valid_assets if (relevant_assets is None) else relevant_assets
	assert(set(relevant_assets) <= set(valid_assets))
	logging.info('running on the following asset(s): ' +', '.join(relevant_assets))

	gs = GridSearchCV(estimator=pipeline, param_grid=grid, scoring='accuracy', cv=cv_splitter, n_jobs=-1, iid=False)

	# XXX Parallelize
	for asset in relevant_assets:
		logging.info('asset: ' +str(asset))

		for ret_ser_name, lab_df in gen_label_dfs(labels, labels_paths, asset):
			logging.info('OG return series: ' +ret_ser_name)
			rep_list = []
			
			for lab_col_name in lab_df:
				logging.info(lab_col_name)
				lab_col_shf_df = lab_df[[lab_col_name]].dropna().shift(periods=-1, freq=None, axis=0).dropna().astype(int)
				prior_ser = lab_col_shf_df[lab_col_name].value_counts(normalize=True, sort=True)

				for one_feat_df in gen_split_feats(features, features_paths, asset):
					lab_feat_df = inner_join(lab_col_shf_df, one_feat_df)
					feat_arr = lab_feat_df.iloc[:, 1:].values
					label_arr = lab_feat_df.iloc[:, 0].values
					assert(feat_arr.shape[0] == label_arr.shape[0])
					res = gs.fit(feat_arr, label_arr)

					logging.info('best_score: ' +str(res.best_score_))
					logging.info('best_params: ' +str(res.best_params_))

					row = {
						'label_name': lab_col_name,
						'feature_name': one_feat_df.columns[0][:-2],
						'best_score': res.best_score_,
						'best_params': res.best_params_,
						'best_index': res.best_index_,
						'adv': prior_ser.max()-res.best_score_
					}
					rep_list.append(row)

			rep_df = pd.DataFrame(rep_list, columns=['label_name', 'feature_name', 'best_score', 'best_params', 'best_index', 'adv'])
			dump_df(rep_df, ret_ser_name, dir_path=RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv') # this is just temporary

if __name__ == '__main__':
	with benchmark('ttf ') as b:
		search_hyperspace(sys.argv[1:])