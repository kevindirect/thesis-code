"""
Kevin Patel
"""

class CategoricalClassifierExperiment(ClassifierExperiment):
	"""Abstract Base Class of all categorical (multi class) classifiers."""

	def __init__(self, other_space={}):
		default_space = {
			'output_activation' : hp.choice('output_activation', ['softmax']),
			'loss': hp.choice('loss', [losses.categorical_crossentropy, losses.sparse_categorical_crossentropy, losses.categorical_hinge, losses.cosine_proximity])
		}
		super(CategoricalClassifierExperiment, self).__init__({**default_space, **other_space})

	def make_const_data_objective(self, features, labels, retain_holdout=True, shuffle=False, test_ratio=.25):
		"""
		Return an objective function that hyperopt can use for the given features and labels.
		"""
		feat_train, feat_test, lab_train, lab_test = get_train_test_split(features, pd.get_dummies(labels, drop_first=False), test_ratio=test_ratio, shuffle=shuffle)

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