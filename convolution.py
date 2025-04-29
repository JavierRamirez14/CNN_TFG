import numpy as np
from scipy import signal
from capa import Capa

class Convolution(Capa):
    def __init__(self, input_shape, kernel_size, n_kernels):
        """
        Capa convolucional.
        
        Args:
            input_shape (tuple): Forma de la entrada (canales, altura, ancho).
            kernel_size (int): Tamaño del kernel (usará kernel_size x kernel_size).
            n_kernels (int): Número de filtros/kernels a aplicar.
        """
        depth, height, width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.n_kernels = n_kernels
        self.output_shape = (n_kernels, height - kernel_size + 1, width - kernel_size + 1)
        self.kernels_shape = (n_kernels, depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)  # Kernels aleatorios
        self.biases = np.random.randn(*self.output_shape)    # Biases aleatorios

    def forward(self, input):
        """Realiza convolución sobre cada canal de entrada."""
        self.input = input
        batch_size = input.shape[0]
        output = np.repeat(np.expand_dims(self.biases, axis=0), repeats=batch_size, axis=0)

        # Convolución para cada imagen, kernel y canal
        for img in range(batch_size):
            for n in range(self.n_kernels):
                for i in range(self.depth):
                    output[img, n] += signal.correlate2d(input[img, i], self.kernels[n,i], 'valid')
        return output

    def backward(self, input_gradient, learning_rate):
        """Calcula gradientes y actualiza kernels y biases."""
        batch_size = input_gradient.shape[0]
        kernels_gradient = np.zeros(self.kernels_shape)
        output_gradient = np.zeros((batch_size, *self.input_shape))

        # Calcula gradientes para kernels y entrada
        for img in range(batch_size):
            for n in range(self.n_kernels):
                for i in range(self.depth):
                    kernels_gradient[n, i] += signal.correlate2d(self.input[img, i], input_gradient[img, n], 'valid')
                    output_gradient[img, i] += signal.convolve2d(input_gradient[img, n], self.kernels[n, i], 'full')

        # Promedia gradientes y actualiza parámetros
        kernels_gradient /= batch_size
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.mean(input_gradient, axis=0)

        return output_gradient / batch_size  # Gradiente promedio