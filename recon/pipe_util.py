# Kevin Patel

import sys
import os
import logging

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common_util import RECON_DIR
from recon.common import dum


DEFAULT_STEP_TRANSLATOR = {
	"kmeans": KMeans(),
	"onehot": OneHotEncoder(),
	"maxent": LogisticRegression(penalty='l2', fit_intercept=True, intercept_scaling=1),
	"__name": "DEFAULT_STEP_TRANSLATOR"
}


def translate_step(step_name, translator=DEFAULT_STEP_TRANSLATOR):
	val = translator.get(step_name, None)
	if (val is None):
		raise ValueError(step_name, 'does not exist in', translator['__name'])
	else:
		return val


def extract_pipeline(dictionary):
	"""
	Converts a passed pipeline dictionary into a sklearn Pipeline object and parameter grid
	"""
	pipeline_steps = [(step_name, translate_step(step_name)) for step_name in dictionary['steps']]
	logging.debug('pipeline structure: ' +str(pipeline_steps))
	pipeline = Pipeline(steps=pipeline_steps)
	
	return pipeline, dictionary['grid']
