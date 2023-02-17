#                          __     __
#     ____ ___  ____  ____/ /__  / /
#    / __ `__ \/ __ \/ __  / _ \/ /
#   / / / / / / /_/ / /_/ /  __/ /
#  /_/ /_/ /_/\____/\__,_/\___/_/
# model stage common

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))
dum = None

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# MODEL
from os import sep
from functools import partial
from common_util import MODEL_DIR

# OTHER STAGE DEPENDENCIES
from data.common import PROC_NAME, VENDOR_NAME

import torch
import torch.nn as nn
import torch.optim as optim
"""
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.train import RMSPropOptimizer, AdamOptimizer
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2
"""

# PACKAGE CONSTANTS
EXP_DIR = MODEL_DIR +f'exp-{PROC_NAME}-{VENDOR_NAME}' +sep
ASSETS = ('SPX', 'RUT', 'NDX', 'DJI')
ASSETS_STR = ','.join(ASSETS)
TRADING_DAYS = 252
MAX_EPOCHS = 80

# PyTorch
class DistributionNLLLoss(nn.modules.loss._Loss):
	"""
	Negative Log Likelihood from a distribution.
	"""
	__constants__ = ['reduction']

	def __init__(self, size_average=None, reduce=None, reduction: str = 'none') -> None:
		super().__init__(size_average, reduce, reduction)

	def forward(self, pred_dist, target) -> torch.Tensor:
		nll = -pred_dist.log_prob(target) / target.shape[1]

		# Weight loss nearer to prediction time?
		# weight = (torch.arange(nll.shape[1]) + 1).float().to(dev)[None, :]
		# lossprob_weighted = nll / torch.sqrt(weight)  # We want to weight nearer points more
		if (self.reduction == 'mean'):
			nll = nll.mean(dim=0)

		return nll

PYTORCH_ACT1D_LIST = ('lrelu', 'celu', 'prelu', 'selu', 'mish', \
	'relu', 'elu', 'gelu', 'sig', 'tanh', 'tanhshrink', 'softplus', 'softmax', 'logsoftmax')
PYTORCH_ACT_MAPPING = {
	'relu': nn.ReLU,
	'lrelu': nn.LeakyReLU,
	'elu': nn.ELU,
	'celu': nn.CELU,
	'gelu': nn.GELU,
	'prelu': nn.PReLU,
	'selu': nn.SELU,
	'mish': nn.Mish,
	'sig': nn.Sigmoid,
	'tanh': nn.Tanh,
	'tanhshrink': nn.Tanhshrink,
	'softplus': nn.Softplus,
	'softmax': partial(nn.Softmax, dim=-1),
	'logsoftmax': partial(nn.LogSoftmax, dim=-1),
	'softmax2d': nn.Softmax2d
}
PYTORCH_INIT_LIST = ('zeros', 'ones', 'normal', 'orthogonal', \
	'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
PYTORCH_LOSS_MAPPING = {
	# Binary
	'clf-bce': nn.BCELoss,
	'clf-bcel': nn.BCEWithLogitsLoss,
	'clf-sm': nn.SoftMarginLoss,

	# Categorical
	'clf-ce': nn.CrossEntropyLoss,
	'clf-nll': nn.NLLLoss,
	'clf-ml': nn.MultiLabelMarginLoss,
	'clf-mls': nn.MultiLabelSoftMarginLoss,
	'clf-dnll': DistributionNLLLoss,

	# Regression
	'reg-mae': nn.L1Loss,
	'reg-mse': nn.MSELoss,
	'reg-sl1': nn.SmoothL1Loss,
	'reg-dnll': DistributionNLLLoss,
}
PYTORCH_OPT_MAPPING = {
	'rms': optim.RMSprop,
	'r': optim.Rprop,
	'adadelta': optim.Adadelta,
	'adagrad': optim.Adagrad,
	'adam': optim.Adam,
	'adamw': optim.AdamW,
	'adamax': optim.Adamax,
	'radam': optim.RAdam,
	'nadam': optim.NAdam,
	'sgd': optim.SGD,
	'asgd': optim.ASGD
}
PYTORCH_SCH_MAPPING = {
	'cos': optim.lr_scheduler.CosineAnnealingLR,
	'cyc': optim.lr_scheduler.CyclicLR,
	'exp': optim.lr_scheduler.ExponentialLR,
	'lam': optim.lr_scheduler.LambdaLR,
	'mst': optim.lr_scheduler.MultiStepLR,
	'rpl': optim.lr_scheduler.ReduceLROnPlateau,
	'st': optim.lr_scheduler.StepLR
}

# Optuna
OPTUNA_DBNAME = 'trials'
OPTUNA_N_TRIALS = 300
OPTUNA_TIMEOUT = 30

