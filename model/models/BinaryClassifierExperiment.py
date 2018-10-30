"""
Kevin Patel
"""

class BinaryClassifierExperiment(ClassifierExperiment):
	"""Abstract Base Class of all binary classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'output_activation' : hp.choice('output_activation', ['sigmoid', 'exponential', 'elu', 'tanh']),
			'loss': hp.choice('loss', [losses.binary_crossentropy, losses.hinge, losses.squared_hinge, losses.kullback_leibler_divergence])
		}
		super(BinaryClassifierExperiment, self).__init__({**default_space, **other_space})

	def make_const_data_objective(self, features, labels, retain_holdout=True, shuffle=False, test_ratio=.25):
		"""
		Return an objective function that hyperopt can use for the given features and labels.
		"""
		feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, labels, test_ratio=test_ratio, shuffle=shuffle)

		def objective(params):
			"""
			Standard classifier objective function to minimize.
			"""
			try:
				compiled = self.make_model(params, features.shape[0])

				if (retain_holdout):
					results = self.fit_model(params, compiled, feat_train, lab_train, shuffle=shuffle, val_split=test_ratio)
				else:
					results = self.fit_model(params, compiled, feat_train, lab_train, lab_train, lab_test, shuffle=shuffle, val_split=test_ratio)

				return {'loss': results, 'status': STATUS_OK}

			except:
				logging.error('Error ocurred during experiment')
				return {'loss': ERROR_CODE, 'status': STATUS_OK}

		return objective