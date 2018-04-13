# Kevin Patel

import sys
import os
from enum import Enum
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from numba import jit, vectorize
from dask import delayed

from common_util import search_df, get_subset #, pd_to_np
from mutate.common import dum


""" ********** GETTING WEIGHTS ********** """
def getWeights(d, size):
	"""
	Returns weighting for arbitrary differentiation.
	Lopez De Prado, Advances in Financial Machine Learning (p. 79)

	Args:
		d (float): coefficient of differentiation
		size (int): window size

	Returns:
		array of weights
	"""
	w = [1.]

	for k in range(1, size):
		w_ = -w[-1] / k * (d-k+1)
		w.append(w_)
	w = np.array(w[::-1]).reshape(-1, 1)

	return w


def getWeights_FFD(d, thresh, max_size=1000):
	"""
	Returns weighting for arbitrary differentiation, fixed window.

	Args:
		d (float): coefficient of differentiation
		thresh (float): threshold
		max_size (int): max possible window size

	Returns:
		array of weights
	"""
	next_weight = 1.0
	weights = []
	k = len(weights)

	while (abs(next_weight) >= thresh and k < max_size):
		weights.append(next_weight)
		k = len(weights)
		next_weight = -weights[-1] / k * (d-k+1)

	weights = np.array(weights[::-1]).reshape(-1, 1)

	return weights


""" ********** FRACTIONAL DIFFERENTIATION ********** """
def fracDiff(raw_df, d, thresh=.01):
	"""
	Expanding window fractional differentiation.
	Lopez De Prado, Advances in Financial Machine Learning (p. 82)

	Increasing width window, with treatment of NaNs
	Note 1: for thresh=1, nothing is skipped
	Note 2: d can be any positive fractional float, not necessarily bounded [0, 1].
	
	Args:
		raw_df (pd.DataFrame): df of series to differentiate
		d (float): coefficient of differentiation
		thresh (float): threshold

	Returns:
		pd.Dataframe of differentiated series
	"""

	# Compute weights for the longest series
	w = getWeights(d, raw_df.shape[0])
	logging.info("finished computing weights")
	logging.debug(str(w[:-5:-1].T))

	# Determine initial calcs to be skipped based on weight-loss threshold
	w_ = np.cumsum(abs(w))
	w_ /= w_[-1]
	skip = w_[w_>thresh].shape[0]

	# Apply weights to values
	df = {}
	for name in raw_df.columns:
		seriesF = raw_df[[name]].fillna(method='ffill').dropna()
		df_= pd.Series()

		for iloc in range(skip, seriesF.shape[0]):
			loc = seriesF.index[iloc]

			if not np.isfinite(raw_df.loc[loc, name]):
				continue # Exclude NAs

			df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0, 0]
		df[name] = df_.copy(deep=True)
	df = pd.concat(df, axis=1)
	return df


def fracDiff_FFD(raw_df, d, thresh=1e-5):
	"""
	Fixed window fractional differentiation.
	Lopez De Prado, Advances in Financial Machine Learning (p. 83-84)

	Note 1: thresh determines the cut-off weight for the window
	Note 2: d can be any positive fractional float, not necessarily bounded [0, 1].
	
	Args:
		raw_df (pd.DataFrame): df of series to differentiate
		d (float): coefficient of differentiation
		thresh (float): threshold

	Returns:
		pd.Dataframe of differentiated series
	"""

	# Compute weights for the longest series
	w = getWeights_FFD(d, thresh)
	width = len(w) - 1
	logging.info("finished computing weights")
	logging.debug(str(w[:-5:-1].T))

	# Apply weights to values
	df = {}
	for name in raw_df.columns:
		seriesF = raw_df[[name]].fillna(method='ffill').dropna()
		df_= pd.Series()

		for iloc1 in range(width, seriesF.shape[0]):
			loc0 = seriesF.index[iloc1-width]
			loc1 = seriesF.index[iloc1]

			if not np.isfinite(raw_df.loc[loc1, name]):
				continue # Exclude NAs

			df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
		df[name] = df_.copy(deep=True)
	df = pd.concat(df, axis=1)
	return df


""" ********** PLOTTING ********** """
def plotWeights(dRange, nPlots, size):
	"""
	Plot differentiation weights for a range of d values.
	Lopez De Prado, Advances in Financial Machine Learning (p. 79)

	Args:
		dRange (float): range of coefficient of differentiation
		nPlots (int): number of different differentiation values used
		size (int): window size

	Returns:
		None
	"""
	w = pd.DataFrame()

	for d in np.linspace(dRange[0], dRange[1], nPlots):
		w_ = getWeights(d, size=size)
		print(w_)
		w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
		w = w.join(w_, how='outer')

	ax = w.plot()
	ax.set_title('weight changes by differentiation')
	ax.set_xlabel('kth previous value')
	ax.set_ylabel('weight')
	ax.legend(loc='upper left')
	plt.show()
	return


def plot_min_FFD_ser(raw_df, ser_name, num_d=11, thresh=.01):
	"""
	Plot ADF (Augmented Dickey-Fuller) statistic for a range of differentiation coefficients.
	Adapted from 'plotMinFFD':
		Lopez De Prado, Advances in Financial Machine Learning (p. 85)

	Args:
		raw_df (pd.DataFrame): 
		ser_name (String): 

	Returns:
		None
	"""

	out = pd.DataFrame(columns=['adf_stat', 'p_val', 'lags', 'n_obs', '95%_conf', 'corr_coeff'])

	for d in np.linspace(0, 1, num_d):
		diff = fracDiff_FFD(raw_df[[ser_name]], d, thresh=thresh)
		corr = np.corrcoef(raw_df.loc[diff.index, ser_name], diff[ser_name])[0, 1]
		diff = adfuller(diff[ser_name], maxlag=1, regression='c', autolag=None)
		out.loc[d] = list(diff[:4]) + [diff[4]['5%']] + [corr] # with critical value

	out[['adf_stat', 'corr_coeff']].plot(secondary_y='adf_stat')
	plt.axhline(out['95%_conf'].mean(), linewidth=1, color='r', linestyle='dotted')
	return out



if __name__=='__main__':
	plotWeights(dRange=[0,1], nPlots=5, size=6)
	plotWeights(dRange=[1,2], nPlots=5, size=6)
