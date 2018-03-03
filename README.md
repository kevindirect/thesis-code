Kevin Patel


## CRUNCH ##
* Crunch is a (python3) package consisting of a data pipeline for the analysis of hourly financial time series data

## STRUCTURE ##
* Crunch's data pipeline consists of a series of stages; each stage is a (python3) subpackage of crunch
* The crunch root directory contains a **common_util.py** module which contains system settings, defaults, and functions all stages can import

### STAGES ###
* A stage consists of (python3) modules, (json) config files for those modules, and potentially some data or artifacts
* Most modules are runnable as one off scripts on the commandline with the _-m_ flag
* Most of the commandline arguments for most modules are paths to json config files
* All modules, runnable or not, can be imported and used by other modules from other stages
* Every stage has a **common.py** whose only required job is to add crunch's parent directory to the python path
* The common.py script can be used for setting default parameters for the stage, constants, and common code for that stage's modules

### PIPELINE ####
* Stages follow a loosely linear pipeline:
	1. raw: pull in raw data from disparate data sources and engage in some cleaning
	2. data: attach raw data together 
	3. transform: processing, feature engineering, and labelling
	4. eda
	5. model
	6. report

## DESIGN PHILOSOPHY ##
* Modularity
* Rapid Prototyping
* Simple

## CONVENTIONS ##
### IO ###
* All variables or constants ending in '\_dir' must be directory strings
* All directory strings must be delimited and terminated by os.sep
* All path names should be absolute
* IO functions:
	- The first argument of any dump function must be the filename (no default)
	- The second argument of any dump function must be the path to the file (defaults to None)
	- Generally file extensions should be inferred by the function and not added to the filename
	- If the IO function is only for one data format (it's in the function name), extensions must be in the filename
