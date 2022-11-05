
* filters and transforms on features and labels
* pipelines defining sequences of the above

# CV (Cross Validation)
* Contains json files representing cross validation schemes
* Each file contains the name/type of cross validation and the parameters to use
* The idea is to be able to easily change the cross validation method by changing a commandline argument

# GTA (Generic Test Applicator)
* Test contains json files representing specifications for series-to-series tests run by the 'gta' script
	- Examples include correlation and missing data counting
	- Tests can include other tests as json files
* The output matrices of these tests are dumped to the report directory
* The gta script takes in dataset and test filenames as commandline arguments
* Modify gta_util and add json files to the test directory to add more tests

# Dataset
* Contains json files representing datasets
* Each key-value pair represents the name and location of a subset of the data (access util or other dataset)
* All data subset locations must directly or indirectly resolve to an access util
* Standard data subset names (like 'features', 'labels', 'targets', and 'row_masks') are assumed by data processing functions in this stage and downstream
	- Some of the standard subset names may have special meanings ('row_masks' is used to mask indices of data to remove invalid rows)
	- 'gen_group' in the 'dataset_util' module takes in a json dataset and yields groups from the data according to given constraints
* This functionality is used heavily downstream by other stages such as 'model'
