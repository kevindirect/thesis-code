# Kevin Patel

import sys
import os
import logging

from sklearn.pipeline import make_pipeline

from common_util import RECON_DIR
from recon.common import dum


DEFAULT_STEP_TRANSLATOR = {
	"kmeans": KMeans(),
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
	pipe_line = make_pipeline(steps=pipeline_steps)
	
	return pipe_line, extract_pipeline['grid']