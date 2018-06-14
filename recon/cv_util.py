# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.cross_validation import _BaseKFold

from common_util import RECON_DIR
from recon.common import dum


class PurgedKFold(_BaseKFold):
	"""
	Extend KFold class to work with labels that span intervals
	The train is purged of observations overlapping test-label intervals
	Test set is assumed contiguous (shuffle=False), w/o training samples in between

	Lopez De Prado, Advances in Financial Machine Learning (p. 131)
	"""
	def __init__(self, n_splits=3, t1=None, pct_embargo=0.):
		if (not isinstance(t1, pd.Series)):
			raise ValueError('Label Through Dates must be a pd.Series')

		super(PurgedKFold, self).__init_(n_splits, shuffle=False, random_state=None)
		self.t1 = t1
		self.pct_embargo = pct_embargo

	def split (self, X, y=None, groups=None):
		if ((X.index == self.t1.index).sum() != len(self.t1)):
			raise ValueError('X and Through Dates must have the same index')

		indices = np.arange(X.shape[0])
		embargo = int(X.shape[0]*self.pct_embargo)
		test_starts = [(i[0], i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]

		for i, j in test_starts:
			t0 = self.t1.index[i] # start of test set
			test_indices = indices[i:j]
			max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
			train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
			if (max_t1_idx < X.shape[0]): # right train (with embargo)
				train_indices = np.concatenate((train_indices, indices[max_t1_idx+embargo:]))

			yield train_indices, test_indices
