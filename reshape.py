import numpy as np
from capa import Capa

class Reshape(Capa):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward(self, input):
        self.batch_size = input.shape[0]
        return np.reshape(input, (self.batch_size, self.output_shape))
    
    def backward(self, input_gradient, learning_rate):
        return np.reshape(input_gradient, (self.batch_size, *self.input_shape))