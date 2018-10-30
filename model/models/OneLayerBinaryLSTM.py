"""
Kevin Patel
"""

class OneLayerBinaryLSTM(BinaryClassifierExperiment):
	"""One layer binary classifier LSTM."""

	def __init__(self, other_space={}):
		default_space = {
			'layer1_size': hp.choice('layer1_size', [8, 16, 32, 64, 128]),
			'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear'])
		}
		super(OneLayerBinaryLSTM, self).__init__({**default_space, **other_space})

	def make_model(self, params, input_shape):
		main_input = Input(shape=input_shape, name='main_input')
		x = LSTM(params['layer1_size'], activation=params['activation'])(main_input)
		output = Dense(1, activation = params['output_activation'], name='output')(x)
		final_model = Model(inputs=[main_input], outputs=[output])
		opt = params['opt'](lr=params['lr'])
		model = final_model.compile(optimizer=opt, loss=params['loss'])

		return model
