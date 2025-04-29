import numpy as np
from capa import Capa

class Dense(Capa):
    def __init__(self, n_input, n_output):
        """
        Capa densa (fully connected).
        
        Args:
            n_input (int): Número de neuronas en la capa de entrada.
            n_output (int): Número de neuronas en la capa de salida.
        """
        # Inicialización Xavier/Glorot para los pesos
        limit = np.sqrt(6 / (n_input + n_output))
        self.weights = np.random.uniform(-limit, limit, (n_input, n_output))
        self.bias = np.zeros((1, n_output))  # Inicialización de bias a cero

    def forward(self, input):
        """Calcula la salida de la capa densa: input * weights + bias."""
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, input_gradient, learning_rate):
        """Propaga el gradiente hacia atrás y actualiza pesos/bias."""
        # Gradiente de los pesos (transpuesta de input * gradiente de entrada)
        weights_gradient = np.dot(self.input.T, input_gradient)
        # Gradiente para la capa anterior (gradiente de entrada * transpuesta de weights)
        output_gradient = np.dot(input_gradient, self.weights.T)
        # Gradiente del bias (suma sobre el batch)
        bias_gradient = np.sum(input_gradient, axis=0, keepdims=True)

        # Actualización de parámetros
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return output_gradient