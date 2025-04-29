import numpy as np
from capa import Capa

class ReLU(Capa):
    def forward(self, input):
        """Aplica ReLU: max(0, input)."""
        self.input = input
        return np.maximum(0, self.input)
    
    def backward(self, output_gradient, learning_rate):
        """Gradiente de ReLU: 1 si input > 0, 0 en otro caso."""
        return (self.input > 0) * output_gradient
    
class Sigmoid(Capa):
    def forward(self, input):
        """Aplica sigmoide: 1/(1 + e^-input)."""
        self.input = input
        return 1 / (1 + np.exp(-self.input))
    
    def backward(self, input_gradient, learning_rate):
        """Gradiente de sigmoide: sigmoid(x)*(1-sigmoid(x))."""
        s = self.forward(self.input)
        return s * (1 - s) * input_gradient
    
class Softmax(Capa):
    def forward(self, input):
        """Aplica softmax con estabilidad numérica."""
        self.input = input
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output
    
    def backward(self, input_gradient, learning_rate):
        """Gradiente de softmax."""
        batch_size = self.output.shape[0]
        grad = np.zeros_like(self.output)
        
        for i in range(batch_size):
            single_output = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            grad[i] = np.dot(jacobian, input_gradient[i])
        
        return grad