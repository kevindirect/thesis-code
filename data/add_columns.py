# Kevin Patel

import pandas as pd
import numpy as np
import datetime


def make_time_cols(df):
	time_cols = pd.DataFrame(index=df.index)
	time_cols['date'] = df.index.map(lambda w: str(w[:10]))
	time_cols['hour'] = df.index.map(lambda w: int(w[11:13]))
	time_cols['dow'] = time_cols['date'].map(lambda w: datetime.datetime.strptime(w, '%Y-%m-%d').weekday())
	time_cols['day'] = df.index.map(lambda w: int(w[8:10]))
	time_cols['month'] = df.index.map(lambda w: int(w[5:7]))
	time_cols['year'] = df.index.map(lambda w: int(w[:4]))

	return time_cols

def make_label_cols(df):
	label_cols = pd.DataFrame(index=df.index)

	# Make price label columns
	label_cols['ret_simple_oc'] = (df['pba_close'] / df['pba_open']) - 1
	label_cols['ret_simple_oa'] = (df['pba_avgPrice'] / df['pba_open']) - 1
	label_cols['ret_simple_oo'] = df['pba_open'].pct_change()
	label_cols['ret_simple_cc'] = df['pba_close'].pct_change()
	label_cols['ret_simple_aa'] = df['pba_avgPrice'].pct_change()
	label_cols['ret_simple_hl'] = (df['pba_high'] / df['pba_low']) - 1
	label_cols['ret_dir_oc'] = np.sign(label_cols['ret_simple_oc'])
	label_cols['ret_dir_oa'] = np.sign(label_cols['ret_simple_oa'])
	label_cols['ret_dir_oo'] = np.sign(label_cols['ret_simple_oo'])
	label_cols['ret_dir_cc'] = np.sign(label_cols['ret_simple_cc'])
	label_cols['ret_dir_aa'] = np.sign(label_cols['ret_simple_aa'])

	# Make volatility label columns
	label_cols['ret_vol_simple_oc'] = (df['vol_close'] / df['vol_open']) - 1
	label_cols['ret_vol_simple_oa'] = (df['vol_avgPrice'] / df['vol_open']) - 1
	label_cols['ret_vol_simple_oo'] = df['vol_open'].pct_change()
	label_cols['ret_vol_simple_cc'] = df['vol_close'].pct_change()
	label_cols['ret_vol_simple_aa'] = df['vol_avgPrice'].pct_change()
	label_cols['ret_vol_simple_hl'] = (df['vol_high'] / df['vol_low']) - 1
	label_cols['ret_vol_dir_oc'] = np.sign(label_cols['ret_vol_simple_oc'])
	label_cols['ret_vol_dir_oa'] = np.sign(label_cols['ret_vol_simple_oa'])
	label_cols['ret_vol_dir_oo'] = np.sign(label_cols['ret_vol_simple_oo'])
	label_cols['ret_vol_dir_cc'] = np.sign(label_cols['ret_vol_simple_cc'])
	label_cols['ret_vol_dir_aa'] = np.sign(label_cols['ret_vol_simple_aa'])

	return label_cols
