"""
Kevin Patel
"""
import sys
import os
from functools import partial
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from numba import jit, vectorize
from dask import delayed

from common_util import isnt, is_df, is_ser, search_df, get_subset
from mutate.common import dum


""" ********** GETTING WEIGHTS ********** """
def getWeights(d, size):
	"""
	Return weighting for arbitrary differencing.
	Number of weights are pre-determined based on size.
	Adapted from:
		Lopez De Prado, Advances in Financial Machine Learning (p. 79)

	Args:
		d (float ∈ ℝ≥0): coefficient of differencing
		size (int ∈ ℤ>0): window size

	Returns:
		array of weights
	"""
	weights = [1.]

	for weight_idx in range(1, size):
		# Calculate next_weight: wₖ = -wₖ₋₁ * ((d-k+1) / k)
		weight_multiplier = (d-weight_idx+1) / weight_idx
		next_weight = -weights[-1] * weight_multiplier
		weights.append(next_weight)

	weights = np.array(weights[::-1]).reshape(-1, 1)
	logging.debug('weights: ' +str(weights.T))

	return weights

def getWeights_FFD(d, thresh):
	"""
	Return weighting for arbitrary differencing, fixed window.
	The number of weights returned are determined based on cutoff threshold.
	Adapted from:
		Lopez De Prado, Advances in Financial Machine Learning (p. 83)

	Args:
		d (float ∈ ℝ≥0): coefficient of fractional differencing
		thresh (float ∈ ℝ>0): threshold

	Returns:
		array of weights
	"""
	weights, weight_idx, next_weight = [], 0, 1 # First weight is one (the latest value which will be differenced)

	while (abs(next_weight) >= thresh):
		weights.append(next_weight); weight_idx += 1;

		# Calculate next_weight: wₖ = -wₖ₋₁ * ((d-k+1) / k)
		weight_multiplier = (d-weight_idx+1) / weight_idx
		next_weight = -weights[-1] * weight_multiplier

	weights = np.array(weights[::-1]).reshape(-1, 1)
	logging.debug('weights: ' +str(weights.T))

	return weights

def getWeights_LB(d, size, thresh):
	"""
	Returns weighting for arbitrary differencing.
	Uses the lower bound of expanding window and fixed window fracdiff.

	Args:
		d (float ∈ ℝ≥0): coefficient of fractional differencing
		size (int ∈ ℤ>0): window size
		thresh (float ∈ ℝ>0): threshold of window cutoff

	Returns:
		array of weights
	"""
	weights, weight_idx = [1.], 1

	while (weight_idx < size):
		# Calculate next_weight: wₖ = -wₖ₋₁ * ((d-k+1) / k)
		weight_multiplier = (d-weight_idx+1) / weight_idx
		next_weight = -weights[-1] * weight_multiplier

		if (abs(next_weight) < thresh):
			break
		else:
			weights.append(next_weight)
			weight_idx += 1

	weights = np.array(weights[::-1]).reshape(-1, 1)
	logging.debug('weights: ' +str(weights.T))

	return weights

def get_weights(d, size=None, thresh=None):
	"""
	Returns weighting for arbitrary differencing.
	If only size is specified, runs fracdiff.
	If only thresh is specified, runs fixed window fracdiff.
	If both size and thresh are specified, uses the lower bound weights.

	Args:
		d (float ∈ ℝ≥0): coefficient of differencing
		size (int ∈ ℤ>0): window size (if None, uses fixed window)
		thresh (float ∈ ℝ>0): threshold of window cutoff (if None, uses expanding window)

	Returns:
		array of weights
	"""
	is_size, is_thresh = size is not None, thresh is not None
	if (not (is_size or is_thresh)): raise ValueError('must specify size and/or thresh')
	weight_fun = {
		is_size and isnt(thresh):	partial(getWeights, d=d, size=size),
		isnt(size) and is_thresh:	partial(getWeights_FFD, d=d, thresh=thresh),
		is_size and is_thresh:		partial(getWeights_LB, d=d, size=size, thresh=thresh)
	}.get(True, None)

	return weight_fun()


""" ********** FRACTIONAL DIFFERENTIATION ********** """
def fracDiff_EFD(raw_df, d, thresh=.01):
	"""
	Expanding window fractional differencing.
	Lopez De Prado, Advances in Financial Machine Learning (p. 82)

	Increasing width window, with treatment of NaNs
	Note 1: for thresh=1, nothing is skipped
	Note 2: d can be any positive fractional float, not necessarily bounded within [0, 1].

	Args:
		raw_df (pd.DataFrame): df of series to differentiate
		d (float): coefficient of differencing
		thresh (float): threshold

	Returns:
		pd.Dataframe of differenced series
	"""

	# Compute weights for the longest series
	w = getWeights(d, raw_df.shape[0])
	logging.debug("finished computing weights: " +str(w[:-5:-1].T))

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
	Fixed window fractional differencing.
	Lopez De Prado, Advances in Financial Machine Learning (p. 83-84)

	Note 1: thresh determines the cut-off weight for the window
	Note 2: d can be any positive fractional float, not necessarily bounded within [0, 1].

	Args:
		raw_df (pd.DataFrame): df of series to differentiate
		d (float): coefficient of differencing
		thresh (float): threshold

	Returns:
		pd.Dataframe of differenced series
	"""

	# Compute weights for the longest series
	w = getWeights_FFD(d, thresh)
	width = len(w) - 1
	logging.debug("finished computing weights: " +str(w[:-5:-1].T))

	# Apply weights to values
	df = {}
	for name in raw_df.columns:
		seriesF = raw_df[[name]].fillna(method='ffill').dropna()
		df_= pd.Series()

		for iloc1 in range(width, seriesF.shape[0]):
			loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]

			if not np.isfinite(raw_df.loc[loc1, name]):
				continue # Exclude NAs

			df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
		df[name] = df_.copy(deep=True)
	df = pd.concat(df, axis=1)
	return df


def fracdiff(raw_df, d, size=None, thresh=1e-5):
	"""
	Fractional differencing

	Note 1: thresh determines the cut-off weight for the window
	Note 2: d can be any positive fractional float, not necessarily bounded within [0, 1].

	Args:
		raw_df (pd.DataFrame): df of series to differentiate
		d (float): coefficient of differencing
		thresh (float): threshold for weights for FFD
		size (int): max size (for Lower Bound FFD) or size (for regular fractional differencing)

	Returns:
		pd.Dataframe of differenced series
	"""
	# Compute weights for the longest series
	w = get_weights(d, size, thresh)
	width = len(w) - 1
	logging.debug("finished computing weights: " +str(w[:-5:-1].T))

	# Apply weights to values
	df = {}
	for name in raw_df.columns:
		seriesF = raw_df[[name]].fillna(method='ffill').dropna()
		df_= pd.Series()

		for iloc1 in range(width, seriesF.shape[0]):
			loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]

			if not np.isfinite(raw_df.loc[loc1, name]):
				continue # Exclude NAs

			df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
		df[name] = df_.copy(deep=True)
	df = pd.concat(df, axis=1)
	return df


""" ********** PLOTTING ********** """
def plotWeights(dRange, nPlots, size):
	"""
	Plot differencing weights for a range of d values.
	Lopez De Prado, Advances in Financial Machine Learning (p. 79)

	Args:
		dRange (float): range of coefficient of differencing
		nPlots (int): number of different differencing values used
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
	ax.set_title('weight changes by differencing')
	ax.set_xlabel('kth previous value')
	ax.set_ylabel('weight')
	ax.legend(loc='upper left')
	plt.show()
	return


def plot_min_FFD_ser(raw_df, ser_name, num_d=11, thresh=.01):
	"""
	Plot ADF (Augmented Dickey-Fuller) statistic for a range of differencing coefficients.
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


def grid_fracdiff_pd(pd_obj, num=100, thresh=.01, size=None):
	"""
	Get ADF (Augmented Dickey-Fuller) statistic for a range of differencing coefficients.

	Args:
		pd_obj (pd.Series|pd.DataFrame): data to difference
		num (int>0): number of differencing coefficients to test. Coefficients are in range (0, 1)
		thresh (float): threshold for weight value
		size (int, optional): maximum number of weights

	Returns:
		MultiIndexed pd.DataFrame
	"""
	pd_df = pd_obj.to_df() if (is_ser(pd_obj)) else pd_obj
	out = pd.DataFrame(columns=['name', 'd', 'adf_stat', 'p_val', 'lags', 'n_obs', '95%_conf', 'corr_coeff']).set_index(['name', 'd'])

	for d in np.linspace(0, 1, num+1):
		diff_df = fracdiff(pd_df, d, size=size, thresh=thresh)
		corr_ser = pd_df.corrwith(diff_df)
		for name in diff_df.columns:
			adf_df = adfuller(diff_df.loc[:, name], maxlag=1, regression='c', autolag=None)
			out.loc[(name, d), :] = list(adf_df[:4]) + [adf_df[4]['5%']] + [corr_ser.loc[name]] # with critical value

	return out


if __name__=='__main__':
	plotWeights(dRange=[0,1], nPlots=5, size=6)
	plotWeights(dRange=[1,2], nPlots=5, size=6)
