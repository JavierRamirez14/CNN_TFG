import numpy as np
from capa import Capa

class Pooling(Capa):
    def __init__(self, kernel_size, stride):
        """
        Versión optimizada de la capa de max pooling.
        
        Args:
            kernel_size (int): Tamaño de la ventana de pooling (cuadrada).
            stride (int): Paso de desplazamiento de la ventana (usualmente igual a kernel_size).
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_shape = None
        self.output_shape = None
        self.indices = None  # Para almacenar los índices de los máximos (útil en la retropropagación)

    def forward(self, input):
        """Aplica la operación de max pooling sobre la entrada."""
        self.input = input
        batch_size, depth, height, width = input.shape
        self.input_shape = input.shape

        # Calcular las dimensiones de la salida
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        self.output_shape = (batch_size, depth, out_height, out_width)

        # Inicializar la salida y los índices de los máximos
        output = np.zeros(self.output_shape)
        self.indices = np.zeros(self.output_shape, dtype=object)

        # Aplicar max pooling
        for img in range(batch_size):  # Iterar sobre cada imagen en el batch
            for d in range(depth):  # Iterar sobre cada canal (profundidad)
                for i in range(out_height):
                    for j in range(out_width):
                        # Definir la ventana de pooling
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Extraer la ventana
                        window = input[img, d, h_start:h_end, w_start:w_end]

                        # Aplicar max pooling y guardar el índice del máximo
                        output[img, d, i, j] = np.max(window)
                        self.indices[img, d, i, j] = np.unravel_index(np.argmax(window), window.shape)

        return output

    def backward(self, input_gradient, learning_rate):
        """Realiza la retropropagación para la capa de pooling."""
        batch_size, depth, out_height, out_width = input_gradient.shape
        output_gradient = np.zeros(self.input_shape)

        # Propagación hacia atrás del gradiente
        for img in range(batch_size):  # Iterar sobre cada imagen en el batch
            for d in range(depth):  # Iterar sobre cada canal (profundidad)
                for i in range(out_height):
                    for j in range(out_width):
                        # Obtener las coordenadas del máximo en la ventana original
                        h_idx, w_idx = self.indices[img, d, i, j]

                        # Calcular las coordenadas en la imagen original
                        h_start = i * self.stride
                        w_start = j * self.stride

                        # Asignar el gradiente al máximo en la ventana original
                        output_gradient[img, d, h_start + h_idx, w_start + w_idx] = input_gradient[img, d, i, j]

        return output_gradient