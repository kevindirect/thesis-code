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

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# MODEL
from os import sep
from functools import partial
from common_util import RECON_DIR, MODEL_DIR

# OTHER STAGE DEPENDENCIES
import torch
import torch.nn as nn
import torch.optim as optim
"""
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.train import RMSPropOptimizer, AdamOptimizer
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2
"""

# PACKAGE CONSTANTS
FR_DIR = MODEL_DIR +'fr' +sep
XG_DIR = MODEL_DIR +'xg' +sep
EXP_LOG_DIR = MODEL_DIR +'exp-log' +sep
EXP_PARAMS_DIR = MODEL_DIR +'exp-params' +sep
XG_PROCESS_DIR = MODEL_DIR +'xg-process' +sep
XG_DATA_DIR = MODEL_DIR +'xg-data' +sep
XG_VIZ_DIR = MODEL_DIR +'xg-graphs-viz' +sep
DATASET_DIR = RECON_DIR +'dataset' +sep
FILTERSET_DIR = RECON_DIR +'filterset' +sep
XG_INDEX_FNAME = '.index.json'
EXPECTED_NUM_HOURS = 8
INTRADAY_LEN = 8
ASSETS = ('sp_500', 'russell_2000', 'nasdaq_100', 'dow_jones')
ASSETS_STR = ','.join(ASSETS)
INTERVAL_YEARS = (2007, 2018)
TRADING_DAYS = 252
WIN_SIZE = 10
TRAIN_RATIO = .5
# VAL_RATIO, TEST_RATIO = .2, .2
# HOPT_WORKER_BIN = 'hyperopt-mongo-worker'
# ERROR_CODE = 999999

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

PYTORCH_MODELS_DIR = MODEL_DIR +'model_p' +sep
PYTORCH_ACT1D_LIST = ('lrelu', 'celu', 'prelu', 'selu', \
	'relu', 'elu', 'gelu', 'sig', 'tanh', 'splus', 'smax', 'logsmax')
PYTORCH_ACT_MAPPING = {
	'relu': nn.ReLU,
	'lrelu': nn.LeakyReLU,
	'elu': nn.ELU,
	'celu': nn.CELU,
	'gelu': nn.GELU,
	'prelu': nn.PReLU,
	'selu': nn.SELU,
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
}
PYTORCH_OPT_MAPPING = {
	'rmsp': optim.RMSprop,
	'adam': optim.Adam
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
OPTUNA_DB_FNAME = 'trials.db'
OPTUNA_CSV_FNAME = 'trials.csv'
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT_HOURS = 12

"""
# Keras
KERAS_MODELS_DIR = MODEL_DIR +'model_k' +sep
KERAS_OPT_MAPPING = {
	'SGD': SGD,
	'RMSprop': RMSprop,
	'Adam': Adam,
	'Nadam': Nadam
}
"""

"""
# TensorFlow
TENSORFLOW_MODELS_DIR = MODEL_DIR +'model_t' +sep
TENSORFLOW_OPT_MAPPING = {
	'RMSprop': RMSPropOptimizer,
	'Adam': AdamOptimizer
}
TENSORFLOW_LOSS_MAPPING = {
	'sparse_ce': sparse_softmax_cross_entropy_with_logits, 	# This version takes in a 1D series of integer labels
	'onehot_ce': softmax_cross_entropy_with_logits_v2	    # One hot encoded version of sparse_cross_entropy
}
"""

# PACKAGE DEFAULTS
dum = None
# default_ray_config = {
# 	"init": {
# 		"num_cpus": 8,
# 		"num_gpus": 1,
# 		"redirect_output": True,
# 		"include_webui": False
# 	}
# }
# default_ray_trial_resources = {"cpu": 2, "gpu": 1}
# default_model = 'TCN'
# default_xg = 'xg0_reteod_direod.json'
# default_dataset = 'xg0_reteod_direod.json'
# default_backend = 'pytorch'
# default_trials_count = 100
# default_filter = ["0"]
# default_nt_filter = ["1"]
# default_opt_filter = ["1", "2"]
# default_target_col_idx = 0
# default_target_idx = [0, 1, 2]


# PACKAGE UTIL FUNCTIONS

