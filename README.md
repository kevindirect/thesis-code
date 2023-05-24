# Thesis Code
## Thesis Details
* Title: An Empirical Evaluation of Neural Process Meta-Learners for Financial Forecasting
* Author: Kevin Patel

## Running
* dump basic preprocessed data: `data/Preproc/run.sh`
* ANP volatiltiy model nowcast hyperparam tuning over train/val: `run-exp-000.bash`
* ANP volatiltiy model forecast hyperparam tuning over train/val: `run-exp-001.bash`
* ANP volatiltiy model forecast (fixed hyperparams) over train/val/test: `run-exp-002.bash`
* simulated trading model over train/val: `model/TradingModel/run.sh`
* simulated trading model over train/val/test: `model/TradingModel/run-final.sh`
* basic smoketests: `smoke-*.sh`

## Overview
* This branch, `master`/`data-frd-minutely`, has all the code submitted for the thesis
* The branch `data-tr-hourly` is for a previous iteration of the project
* Unfortunately raw data used to generate the results was purchased from FirstRate Data LLC, so it cannot be shared

### `data/`
* has all the raw and preprocessed data
* preprocessing is done from Julia scripts/Pluto.jl notebooks in the Preproc package
* preprocessed data can be loaded in Python via a PytorchLightning DataModule

### `model/`
* has all the model code (Pytorch models wrapped in PytorchLightning)

#### RV Model
* `{model, np}_util.py` contain Pytorch model classes
* `pl_{generic, np}.py` are LightningModule classes that wrap Pytorch models for PytorchLightning
* `expo.py` is the optuna hyperparameter optimizing runner
* `expm.py` is the manual/fixed hyperparameter experiment runner
* the `model/exp-<proc>-<data>` directories contain completed realized volatility trial results
* hyperparameter sets are stored on disk in json files

#### Trading Model
* `model/TradingModel` contains the julia package with the trading model code
* Trading model uses features/predictions dumped from the last realized volatiltiy models
* `model/tm-<proc>-<data>` directories contain completed trading model results

### `common_util.py`
* `common_util.py` contains functions, data structures, and classes used throughout the project
* Much of the code in here is not used anymore (previous iterations of project)

### Other
* Most code is arranged in python subpackages, `<subpackage>/common.py` contains common constants, defaults, and utilities
* Subpackage scripts are run by running them as modules (using the `-m` flag), see the shell scripts at the project root for examples
