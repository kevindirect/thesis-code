# Crunch

## Overview
### common_util
* `common_util.py` contains functions, data structures, and classes used throughout the project. Most of the code in here is probably not used anymore
* Most code aside from `common_util.py` is arranged in python subpackages, `<subpackage>/common.py` contains common constants, defaultsi, and utilities
* Subpackage scripts are run by running them as modules (using the `-m` flag), see the shell scripts at the project root for examples
* There is a Julia package that contains basic data preprocessing code

### data
* has all the raw and preprocessed data
* preprocessing is done from Julia scripts/Pluto.jl notebooks in the Preproc package
* preprocessed data can be loaded in Python via a PytorchLightning DataModule

### model
* has all the model code (Pytorch models wrapped in PytorchLightning)
* `{model, np}_util.py` contain Pytorch model classes
* `pl_{generic, np}.py` are LightningModule classes that wrap Pytorch models for PytorchLightning
* `exp.py` is the main experiment script.
* the `model/exp-<proc>-<data>` directories contain completed trial results
* hyperparameter sets are stored on disk in json files

### recon
* contains some miscellaneous modules
* most of these have been deperecated since earlier versions of the project
* main relevant module is `recon/viz.py` which contains plotting functions

### deprecated subpackages
* contains lots of deprecated code/subpackages from earlier versions of the project, there is lots and lots of junk here
* see other branches of the repo for a lot more deprecated code

## TODO
* test new data with params
* Do some param tuning to see how model works with the new data and target
* try out a few different rvol targets
* output graphs with mean + sd for use with latent variable models
* dump a simulated trading model (buy&hold with _volatility avoidance_)
* (optional) discount rate

