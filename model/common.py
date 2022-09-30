#                          __     __
#     ____ ___  ____  ____/ /__  / /
#    / __ `__ \/ __ \/ __  / _ \/ /
#   / / / / / / /_/ / /_/ /  __/ /
#  /_/ /_/ /_/\____/\__,_/\___/_/
# model stage common
"""
Kevin Patel
"""

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
from data.common import PROC_NAME, DATA_NAME

import torch
import torch.nn as nn
import torch.optim as optim
"""
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.train import RMSPropOptimizer, AdamOptimizer
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2
"""

# PACKAGE CONSTANTS
EXP_DIR = MODEL_DIR +f'exp-{PROC_NAME}-{DATA_NAME}' +sep
# EXP_LOG_DIR = MODEL_DIR +'log' +sep
# EXP_PARAMS_DIR = MODEL_DIR +'params' +sep
ASSETS = ('SPX', 'RUT', 'NDX', 'DJI')
ASSETS_STR = ','.join(ASSETS)
TRADING_DAYS = 252

# PyTorch
class DistributionNLLLoss(nn.modules.loss._Loss):
	"""
	Negative Log Likelihood of getting a value from a distribution.
	"""
	__constants__ = ['reduction']

	def __init__(self, size_average=None, reduce=None, reduction: str = 'none') -> None:
		super().__init__(size_average, reduce, reduction)

	def forward(self, pred_dist, target) -> torch.Tensor:
		# if (type(out_dist).__name__ in ('Bernoulli', 'Beta', 'Normal', 'LogNormal')):
		# 	ftype = {
		# 		16: torch.float16,
		# 		32: torch.float32,
		# 		64: torch.float64
		# 	}.get(cast_precision, 16)
		# 	label_y = label_y.to(ftype)
		# 	if (type(out_dist).__name__ in ('Beta',)):
		# 		eps = 1e-3
		# 		label_y = label_y.clamp(min=eps, max=1-eps)
		nll = -pred_dist.log_prob(target)

		# Weight loss nearer to prediction time?
		# weight = (torch.arange(nll.shape[1]) + 1).float().to(dev)[None, :]
		# lossprob_weighted = nll / torch.sqrt(weight)  # We want to weight nearer stuff more
		if (self.reduction == 'mean'):
			nll = nll.mean(dim=0)

		return nll

class SharpeLoss(nn.modules.loss._Loss):
	"""
	Sharpe Ratio based loss
	pred is a floating point bet size * direction, target is the actual return
	"""
	__constants__ = ['reduction']

	def __init__(self, size_average=None, reduce=None, reduction: str = 'none',
		weight=None, go_long=True, go_short=False) -> None:
		super().__init__(size_average, reduce, reduction)
		self.go_long = go_long
		self.go_short = go_short

	def forward(self, pred, target) -> torch.Tensor:
		# if (type(out_dist).__name__ in ('Bernoulli', 'Beta', 'Normal', 'LogNormal')):
		# 	ftype = {
		# 		16: torch.float16,
		# 		32: torch.float32,
		# 		64: torch.float64
		# 	}.get(cast_precision, 16)
		# 	label_y = label_y.to(ftype)
		if (not self.go_long):
			pred[pred > 0.0] = 0.0
		if (not self.go_short):
			pred[pred < 0.0] = 0.0
		ret = pred * target
		sr = torch.tensor(TRADING_DAYS).sqrt() * (ret.mean() / ret.std())
		sr_loss = torch.exp(-sr)

		if (self.reduction == 'mean'):
			sr_loss = sr_loss.mean(dim=0)

		return sr_loss

PYTORCH_ACT1D_LIST = ('lrelu', 'celu', 'prelu', 'selu', 'mish', \
	'relu', 'elu', 'gelu', 'sig', 'tanh', 'splus', 'smax', 'logsmax')
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
	'splus': nn.Softplus,
	'smax': partial(nn.Softmax, dim=-1),
	'logsmax': partial(nn.LogSoftmax, dim=-1),
	'smax2d': nn.Softmax2d
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

	# Other
	'reg-sharpe': SharpeLoss,
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

# # Optuna
# OPTUNA_DB_FNAME = 'trials.db'
# OPTUNA_CSV_FNAME = 'trials.csv'
# OPTUNA_N_TRIALS = 100
# OPTUNA_TIMEOUT_HOURS = 12

