"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import isfile, basename
from functools import partial
import logging

import numpy as np
import pandas as pd
from dask import delayed
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
#from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, ElasticNet
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

from common_util import MODEL_DIR, JSON_SFX_LEN, DF_DATA_FMT, makedir_if_not_exists, is_valid, str_to_list, get_cmd_args, load_json, dump_df, benchmark
from common_util import pairwise, compose, pd_split_ternary_to_binary, df_midx_restack, np_value_counts, pd_rows, midx_intersect, pd_get_midx_level
from model.common import FR_DIR
from model.xg_util import get_xg_feature_dfs, get_xg_label_target_dfs
from model.preproc_util import temporal_preproc_3d, stride_preproc_3d
from model.train_util import pd_to_np_tvt


def fr(argv):
	cmd_arg_list = ['models=', 'assets=', 'window_size=', 'knn=', 'trials_count=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	models = str_to_list(cmd_input['models=']) if (is_valid(cmd_input['models='])) else 'rf'
	assets = str_to_list(cmd_input['assets=']) if (is_valid(cmd_input['assets='])) else ['sp_500', 'russell_2000', 'nasdaq_100', 'dow_jones']
	window_size = int(cmd_input['window_size=']) if (is_valid(cmd_input['window_size='])) else 1
	knn = int(cmd_input['knn=']) if (is_valid(cmd_input['knn='])) else 3
	trials_count = int(cmd_input['trials_count=']) if (is_valid(cmd_input['trials_count='])) else 100
	params = { 'window_size': window_size, 'knn': knn, 'loss': 'ce' }

	for asset in assets:
		fd = get_xg_feature_dfs(asset)
		ld, td = get_xg_label_target_dfs(asset)
		pba_hoc_ddir = pd_split_ternary_to_binary(ld['hoc']['pba']['ddir'].replace(to_replace=-1, value=0))
		pba_hoc_dret = pd_split_ternary_to_binary(td['hoc']['pba']['dret'])

		for freq in ['h', 'd']:
			for src in ['pba', 'vol', 'buzz', 'nonbuzz']:
				dest_dir = FR_DIR +sep.join([asset, freq, src]) +sep
				makedir_if_not_exists(dest_dir)

				for axe, subsets in fd[freq][src].items():
					res, dest_fname = [], '{a}.{m}.w{w}.n{n}'.format(a=axe, m='mi', w=params['window_size'], n=params['knn'])
					if (isfile('{}{}.{}'.format(dest_dir, dest_fname, DF_DATA_FMT))):
						logging.info('skipping {}...'.format(dest_fname))
					else:
						logging.info('running {}...'.format(dest_fname))
						for subset_key, subset_df in subsets.items():
							logging.info(subset_key)

							for ser_name in list(subset_df.index.unique(level='id1')):
								logging.info(ser_name)
								ser_subset_df = subset_df.xs(ser_name, level=1, drop_level=False)
								ser_subset_df.index = ser_subset_df.index.remove_unused_levels()
								flt_train_win, flt_val_win, flt_test_win = data_preproc(ser_subset_df, pba_hoc_ddir, pba_hoc_dret, params)
								# X = np.concatenate((flt_train_win[0], flt_val_win[0], flt_test_win[0]), axis=0)
								# y = np.concatenate((flt_train_win[1], flt_val_win[1], flt_test_win[1]), axis=0)
								# z = np.concatenate((flt_train_win[2], flt_val_win[2], flt_test_win[2]), axis=0)
								for mi in mutual_info(*flt_train_win, params, list(ser_subset_df.columns)):
									res.append((subset_key, ser_name, *mi))
						res_df = pd.DataFrame.from_records(res, columns=['sub', 'ser', 'col', 'clf', 'reg', 'win', 'knn'])
						dump_df(res_df, dest_fname, dir_path=dest_dir)


# Data
def data_preproc(f_df, l_df, t_df, params, year_interval=('2009', '2018')):
	"""
	Top level data preprocessing function.
	"""
	common_idx = midx_intersect(pd_get_midx_level(f_df), pd_get_midx_level(l_df), pd_get_midx_level(t_df))
	common_idx = common_idx[(common_idx > year_interval[0]) & (common_idx < year_interval[1])]
	feature_df, label_df, target_df = map(compose(partial(pd_rows, idx=common_idx), df_midx_restack), [f_df, l_df, t_df])
	assert(all(feature_df.index.levels[0]==label_df.index.levels[0]))
	assert(all(feature_df.index.levels[0]==target_df.index.levels[0]))

	flt_train, flt_val, flt_test = __setup_data__((feature_df, label_df, target_df), params)
	return (__preproc__(flt_train, params), __preproc__(flt_val, params), __preproc__(flt_test, params))


def __setup_data__(data, params):
	"""
	Set self.flt_{train, val, test} by converting (feature_df, label_df, target_df) to numpy dataframes split across train, val, and test subsets.
	"""
	flt_train, flt_val, flt_test = zip(*map(pd_to_np_tvt, data))
	shapes = np.asarray(tuple(map(lambda tvt: tuple(map(np.shape, tvt)), (flt_train, flt_val, flt_test))))
	assert all(np.array_equal(a[:, 1:], b[:, 1:]) for a, b in pairwise(shapes)), 'feature, label, target shapes must be identical across splits'
	assert all(len(np.unique(mat.T[0, :]))==1 for mat in shapes), 'first dimension (N) must be identical length in each split for all (feature, label, and target) tensors'
	return (flt_train, flt_val, flt_test)

def __preproc__(data, params, overlap=True):
	"""
	Window reshape.
	"""
	x, y, z = temporal_preproc_3d(data, window_size=params['window_size'], apply_idx=[0]) if (overlap) else stride_preproc_3d(data, window_size=params['window_size'])
	if (params['loss'] in ('bce', 'bcel', 'ce', 'nll')):
		y_new = np.sum(y, axis=(1, 2), keepdims=False)		# Sum label matrices to scalar values
		z_new = np.sum(z, axis=(1, 2), keepdims=False)		# Sum label matrices to scalar values
		if (y.shape[1] > 1):
			y_new += y.shape[1]				# Shift to range [0, C-1]
		if (params['loss'] in ('bce', 'bcel') and len(y_new.shape)==1):
			y_new = np.expand_dims(y_new, axis=-1)
		y = y_new
		z = z_new
	return (np_collapse_last_two_dim(x), y, z)

def np_collapse_last_two_dim(arr):
	return arr.reshape(arr.shape[0], arr.shape[1]*arr.shape[2])


# Feature Ranking
def mutual_info(X, y, z, params, feat_names):
	mi_clf = mutual_info_classif(X, y, n_neighbors=params['knn'])
	mi_reg = mutual_info_regression(X, z, n_neighbors=params['knn'])
	return [(f, mi_clf[i], mi_reg[i], params['window_size'], params['knn']) for i, f in enumerate(feat_names)]


def feat_imp_mdi(fit, feat_names):
	df0 = {i: tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
	df0 = pd.DataFrame.from_dict(df0, orient='index')
	df0.columns = feat_names
	df0 = df0.replace(0, np.nan) # because max_features=1
	imp = pd.concat({'mean': df0.mean(), 'std': df0.std()*df.shape[0]**-.5}, axis=1)
	imp /= imp['mean'].sum()
	return imp

	#print(flt_train_win[0], flt_val_win[0])
	# RandomForestClassifier
	#clf = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True, max_depth=None,
	#	min_impurity_decrease=0, random_state=0).fit(flt_train_win[0], flt_train_win[1])
	#sc = clf.score(flt_val_win[0], flt_val_win[1])

	#print('base :', flt_val_win[1].mean())
	#print('score:', sc)
	#print('diff : {:%}'.format((sc-flt_val_win[1].mean())))
	#sys.exit(0)

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		fr(sys.argv[1:])
