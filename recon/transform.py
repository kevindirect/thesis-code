# Kevin Patel

import sys
import os
import logging
import numpy as np
import pandas as pd
import engarde.decorators as ed

from common_util import load_df, dump_df, DATA_DIR
from data.data_api import DataAPI



# Constant Threshold Labelling
def signify(ser, thresh):
	"""
	Apply sign function to ser based on a passed thresh
	"""
	return ser

# Volatility Based Threshold Labelling

# Triple Barrier Labelling




@ed.is_shape((None, 2))
def subtract(df):
	return df.iloc[0] - df.iloc[1]

def difference(ser, periods=1):
	return ser.diff(periods=periods)

def sma(ser, window=8, min_periods=None, win_type=None):
	return ser.rolling(window, min_periods=min_periods, win_type=win_type).mean()

def ema(ser, com=None, span=8, halflife=None, alpha=None):
	"""
	One must specify precisely one of span, center of mass, half-life and alpha to the EW functions
	Span corresponds to what is commonly called an “N-day EW moving average”.
	Center of mass has a more physical interpretation and can be thought of in terms of span: c=(s−1)/2.
	Half-life is the period of time for the exponential weight to reduce to one half.
	Alpha specifies the smoothing factor directly.
	"""
	return ser.ewm(window, com=com, span=span, halflife=halflife, alpha=alpha).mean()

print(DataAPI.print())

search_dict = {'group'}