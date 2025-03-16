import numpy as np
from scipy import signal
from capa import Capa

class Convolution(Capa):

    def __init__(self, input_shape, kernel_size, n_kernels):
        depth, height, width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.n_kernels = n_kernels
        self.output_shape = (n_kernels, height - kernel_size + 1, width - kernel_size + 1)
        self.kernels_shape = (n_kernels, depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        output = np.repeat(np.expand_dims(self.biases, axis=0), repeats=batch_size, axis=0)

        for img in range(batch_size):
            for n in range(self.n_kernels):
                for i in range(self.depth):
                    output[img, n] += signal.correlate2d(input[img, i], self.kernels[n,i], 'valid')
        return output

    def backward(self, input_gradient, learning_rate):
        batch_size = input_gradient.shape[0]
        kernels_gradient = np.zeros(self.kernels_shape)
        output_gradient = np.zeros((batch_size, *self.input_shape))

        # print(f'Input gradient shape: {input_gradient.shape} | Kernels gradient shape: {kernels_gradient.shape} | Output gradient shape: {output_gradient.shape}')

        for img in range(batch_size):  # Iterar sobre cada imagen en el mini-batch
            for n in range(self.n_kernels):  # Iterar sobre cada kernel
                for i in range(self.depth):  # Iterar sobre cada canal de la imagen
                    # Calcular el gradiente de los kernels
                    kernels_gradient[n, i] += signal.correlate2d(self.input[img, i], input_gradient[img, n], 'valid')
                    # Calcular el gradiente de la entrada
                    output_gradient[img, i] += signal.convolve2d(input_gradient[img, n], self.kernels[n, i], 'full')

        # Promediar los gradientes sobre el mini-batch
        kernels_gradient /= batch_size
        output_gradient /= batch_size

        # Actualizar los kernels y los biases
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.mean(input_gradient, axis=0)  # Promediar sobre el mini-batch

        return output_gradient