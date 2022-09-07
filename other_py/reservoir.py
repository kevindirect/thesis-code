# Kevin Patel
# EE 509
# 6/5/2017

import numpy as np
import scipy.linalg as la

class Reservoir(object):

    def __init__(self, size=100, connectivity=1, spectral_radius=1, standardize_input=True,
                 input_scaling=1, leaking_rate=.5, discard=0):
        """
        :param size: reservoir size
        :param connectivity: reservoir connectivity ratio
        :param spectral_radius: reservoir spectral radius
        :param standardize_input: boolean to standardize input or not
        :param input_scaling: input scaling for bias and input
        :param leaking_rate: reservoir leaking rate
        :param discard: number of steps to discard before predictions start
        """
        self.size = size
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.standardize_input = standardize_input
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.discard = discard
        self.__init_reservoir__()

    def __str__(self):
        string = 'reservoir size: ' +str(self.size) \
                +'\nconnectivity: {0:.4f}'.format(self.connectivity) \
                +'\nspectral radius: {0:1.4f}'.format(self.spectral_radius) \
                +'\nstandardize input: ' +str(self.standardize_input) \
                +'\ninput scaling: {0:1.4f}'.format(self.input_scaling) \
                +'\nleaking rate: {0:.4f}'.format(self.leaking_rate) \
                +'\ndiscard steps: ' +str(self.discard)
        return string

    def __init_reservoir__(self):
        # Create reservoir
        size_sq = self.size * self.size
        self.weights = np.random.rand(size_sq) - 0.5

        # Sparsify Reservoir
        if (self.connectivity < 1):
            disconnections = np.random.permutation(size_sq)[int(size_sq * self.connectivity * 1.0):]
            self.weights[disconnections] = 0

        # Set spectral radius
        self.weights = self.weights.reshape((self.size, self.size))
        initial_spectral_radius = np.max(np.abs(la.eig(self.weights)[0]))
        self.weights *=  self.spectral_radius / initial_spectral_radius

    # Return reservoir activation states with provided data matrix
    def transform(self, data):
        """
        :param data: data matrix
        """
        def numpy_minmax(X):    # [0, 1] range standardization function across outer dimension
            xmin =  X.min(axis=0)
            return (X - xmin) / (X.max(axis=0) - xmin)

        num_ex = data.shape[0]
        num_dim = data.shape[1]

        if (self.standardize_input):
            data = numpy_minmax(data)
        data = data * self.input_scaling

        input_weights = (np.random.rand(self.size, 1 + num_dim) - 0.5) * 1

        # Init activation state matrix
        activations = np.zeros((num_ex - self.discard, 1 + num_dim + self.size))
        curr_state = np.zeros(self.size)

        # Collect activation states
        for t, vec in enumerate(data):
            past_echo = (1 - self.leaking_rate) * curr_state
            vec_with_bias = np.hstack((1 * self.input_scaling, vec))
            input_factor = np.dot(input_weights, vec_with_bias)
            recurrent_factor = np.dot(self.weights, curr_state)

            curr_state = past_echo + self.leaking_rate * np.tanh(input_factor + recurrent_factor)
            if t >= self.discard:
                # Create activation by appending input vector with reservoir output vector
                activations[t-self.discard] = np.hstack((vec_with_bias, curr_state))
        return activations
