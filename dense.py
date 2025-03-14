import numpy as np
from capa import Capa

class Dense(Capa):

    def __init__(self, n_input, n_output):
        limit = np.sqrt(6 / (n_input + n_output))
        self.weights = np.random.uniform(-limit, limit, (n_input, n_output))
        self.bias = np.zeros((1, n_output))

    def forward(self, input):
        self.input = input
        m = np.dot(self.input, self.weights)
        return m + self.bias
    
    def backward(self, input_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, input_gradient)
        output_gradient = np.dot(input_gradient, self.weights.T)
        bias_gradient = np.sum(input_gradient, axis=0, keepdims=True)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return output_gradient