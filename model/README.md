# XG
* The model stage contains an experiment group (xg) pipeline to do some more processing on the mutate stage data prior to runtime processing.
* XG data are structured as MultiIndex Dataframes of shape (N, C, D)
* The XG_PROCESS_DIR contains json files that describe how the experiment groups are to be processed.
	- 'dataset': Points to a recon dataset json file
	- 'constraint': the 'dataset_util' constraint function to constrain permutations to a valid subset (for instance all subsets must be of the same asset)
	- 'parts': Sets the partitions of data to be extracted
		- An item can be a string or a list
		- If it is a list, all the items in the list are passed to the preprocessing function that is set by the first item in the list
	- 'how': Sets the way parts are yielded to the processing function(s).
	- 'prep_fns': Sets preprocessing functions to be used with datagen on particular data subsets. Possible functions are in dataprep_util.py DATA_PREP_MAPPING
		- If no function is supplied for a subset no preprocessing is done
* The xgpp.py module creates a dask graph to compute and dump all experiment group dataframes to XG_DATA_DIR in parallel (xgp.py does the same process serially). Dask visualizations and test runs are also possible with xgpp.py.
* Experiement groups can be loaded with xgload from xg_util.py

# Runtime Processing
* The nb-model_xg-model-data.ipynb notebook shows the process of runtime processing prior to modelling.
	1. xg data is loaded
	2. relevant subsets are extracted
	3. features are joined (pd.concat) to make a feature DataFrame
	4. common rows are found across features and labels/targets
	5. data is splitinto train/val/test
	6. data is converted to numpy tensors
	7. data is processed from a function preproc_util (ie moving window)
* After this process the features are four dimensional (N-W+1, C, W, D) or (N//W, C, W, D)
* The labels/targets are three (N-W+1, C, D) or four (N//W, C, W, D) dimensional

