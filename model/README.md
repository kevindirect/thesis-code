
# Models
* There are three model backends: keras, pytorch, and tensorflow
* A model consists of a 'hyperopt' hyperparameter space and a function that returns an objective to minimize
* The model can contain last minute preprocessing by implementing a 'preproc' method via a mixin class
* The objective can fix the data (as in 'make_const_data_objective') or treat all the data as another hyperparameter to search over
* The 'hexp' script takes in an experiment group, searches its hyperparameter space, and dumps metrics to the report stage and within a mongo database
	- The 'trials-count' argument sets the number of iterations to search over a hyperparameter space for before the next loop of datagen


# XG (Experiment Groups)
* Packages a recon dataset with other information to run a group of experiments
	- 'dataset': Points to a recon dataset json file
	- 'constraint': the 'dataset_util' constraint function to constrain permutations to a valid subset (for instance all subsets must be of the same asset)
	- 'parts': Sets the partitions of data to be extracted
		- An item can be a string or a list
		- If it is a list, all the items in the list are passed to the preprocessing function that is set by the first item in the list
	- 'prep_fns': Sets preprocessing functions to be used with datagen on particular data subsets
		- If no function is supplied for a subset no preprocessing is done
		- 'common' denotes a common preprocessing function that takes in all datasets, runs pd_common_index_rows by default