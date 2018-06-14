# Kevin Patel

import sys
import os
import logging

from sklearn.model_selection import KFold, TimeSeriesSplit

from common_util import RECON_DIR
from recon.common import dum


DEFAULT_CV_TRANSLATOR = {
	"KMeans": KMeans,
	"TimeSeriesSplit": TimeSeriesSplit,
	"__name": "DEFAULT_CV_TRANSLATOR"
}


def translate_cv(cv_name, cv_params=None, translator=DEFAULT_CV_TRANSLATOR):
	cv_constructor = translator.get(cv_name, None)

	if (cv_constructor is None):
		raise ValueError(cv_name, 'does not exist in', translator['__name'])
	else:
		if (cv_params is not None):
			return cv_constructor(**cv_params)
		else:
			return cv_constructor()


def extract_cv_splitter(dictionary):
	"""
	Converts a passed pipeline dictionary into a sklearn Pipeline object and parameter grid
	"""
	cv_splitter = translate_cv(dictionary['name'], cv_params=dictionary['params'])
	
	return cv_splitter
