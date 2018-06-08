# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_util import benchmark, inner_join
from mutate.label import LABEL_SFX_LEN
from recon.common import dum


def get_base_labels(df_columns):
	return list(set([col_name[:-LABEL_SFX_LEN] for col_name in df_columns]))




def corr_mat(df=None, feat_col_name=None, lab_col_name=None, **kwargs):
	cm = df.corr(**kwargs).loc[feat_col_name, lab_col_name]
	return cm


def corr_matrix(data, features, labels):
	corr = pd.DataFrame()
	for feat in features:
		row = {}
		row['feature'] = feat
		row.update({label: data[feat].corr(data[label]) for label in labels})
		corr = corr.append(row, ignore_index=True)
	return corr

if __name__ == '__main__':
	main(sys.argv[1:])
