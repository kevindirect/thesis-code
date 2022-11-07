"""
Kevin Patel
"""
import sys
import os
import logging
import pickle

import numpy as np
import pandas as pd
import torch
import torchmetrics as tm

from common_util import is_valid, isnt
from model.common import dum


class DictMetric(tm.Metric):
	"""
	Track a dictionary of metrics.
	"""
	def __init__(self, keys, compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		self.keys = keys
		for key in self.keys:
			self.add_state(key, default=[], dist_reduce_fx='cat')

	def update(self, key, val):
		getattr(self, key).append(val)

	def compute(self, key):
		return torch.cat(getattr(self, key), dim=0)

