#    _|_|    _|      _|  _|_|_|_|
#  _|    _|    _|  _|    _|
#  _|_|_|_|      _|      _|_|_|
#  _|    _|    _|  _|    _|
#  _|    _|  _|      _|  _|_|_|_|
# JSON based data access utility.
"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import dirname, basename
from functools import partial
import logging

import numpy as np
import pandas as pd

from common_util import isnt, load_json
from data.common import AXEFILES_DIR, DG_PFX, CS_PFX


# ********** AXE FIELDS **********
AXE_ALL, AXE_SUB = "*", "/"


# ********** AXE EXCEPTION CLASSES **********
class AxeFormatError(Exception):
	"""
	Use this class for formatting/validation errors in axefiles.
	"""
	def __init__(self, message):
		super().__init__(message)


# ********** AXE UTILITIES **********
def axe_get(axe, axe_dir=AXEFILES_DIR):
	"""
	Loads and returns raw axefile from given name.
	"""
	if (not axe[1].startswith(axe[0])):
		error_msg = 'invalid axefile name \'{}\', second string must be substring of the first (view axefile) or identical to the first (root axefile)'.format(str(axe))
		logging.error(error_msg)
		raise AxeFormatError(error_msg)

	rootaxe_dir = axe_dir +axe[0] +sep
	fmt = '{}{}.json'
	axefnames = (fmt.format(DG_PFX, axe[1]), fmt.format(CS_PFX, axe[1]))
	return tuple(map(partial(load_json, dir_path=rootaxe_dir), axefnames))


def axe_process(axe_dict):
	"""
	This function should add the AXE_ALL field value to each of the AXE_SUB values.
	"""
	if (AXE_ALL not in axe_dict or AXE_SUB not in axe_dict):
		error_msg = 'the \'{}\' ({}) and \'{}\' ({}) keys are required to exist'.format(AXE_ALL, 'all', AXE_SUB, 'subsets')
		logging.error(error_msg)
		raise AxeFormatError(error_msg)

	xall = axe_dict[AXE_ALL] or {}
	if (isnt(axe_dict[AXE_SUB])):
		return {None: xall}	# No subsets
	else:
		return {name: {**xall, **d} for name, d in axe_dict[AXE_SUB].items()}


def axe_join(*axefile):
	"""
	Joins an axefile into a dictionary used for querying the data record,
	does some validation on the axefile in the process.

	Args:
		*axefile (iterable): processed representation of an axefile

	Returns:
		Joined axefile (dictionary)
	"""
	if (any([a.keys()!=axefile[0].keys() for a in axefile[1:]])):
		error_msg = 'the files making up the axefile must either all have identical subset names or no subsets'
		logging.error(error_msg)
		raise AxeFormatError(error_msg)
	return {key: tuple(axe[key] for axe in axefile) for key in axefile[0].keys()}


def axe_get_keychain(prefix, axe, suffix, subset=None):
	"""
	Return the axe keychain for the given components.

	This is generally the input lists concatenated in the order they
	are passed in but there is a special case when the axefile is a view axefile and the subset
	does not match the desc field (post variant removal - removing any parentheses and content within
	them - from desc). In these latter cases we swap the subset and suffix keys. This is important because
	the last key is generally seen to be the most finegrained identifier and is used for a lot of processing
	downstream.

	Args:
		prefix (list): currently a singleton containing the 'root' field of the record
		axe (list): a two item list containing the axefile identifier
		suffix (list): currently a singleton containing the 'desc' field of the record
		subset (str|None): subset name

	Returns:
		If there are no subsets, returns a four item list:
			* [prefix, axe[0], axe[1], suffix]
		Exepected case returns a five item list:
			* [prefix, axe[0], axe[1], subset, suffix]
		If a special case is triggered it will swap the subset and suffix items:
			* [prefix, axe[0], axe[1], suffix, subset]
	"""
	if (isnt(subset)):
		keychain = prefix+axe+suffix				# no subset keychain
	else:
		sfx = re.sub('[()]', '', suffix[0]) if ('(' in suffix[0] and ')' in suffix[0]) else suffix[0]

		if (sfx == subset):
			keychain = prefix+axe+[subset]+suffix		# standard keychain
		else:
			if (axe[0]==axe[1]):
				error_msg = 'subset and desc must be identitical for root axefiles (post variant removal from desc)'
				logging.error(error_msg)
				raise AxeFormatError(error_msg)
			elif (axe[1].startswith(axe[0])):
				keychain = prefix+axe+suffix+[subset]	# swap suffix and subset
			else:
				error_msg = 'invalid axefile name \'{}\', second string must be substring of or identical to the first'.format(str(axe))
				logging.error(error_msg)
				raise AxeFormatError(error_msg)
	return keychain


