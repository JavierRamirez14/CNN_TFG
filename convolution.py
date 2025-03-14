import numpy as np
from capa import Capa

class Convolution(Capa):
    def __init__(self, input_shape, kernel_size, n_kernels):
        depth, height, width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.output_shape = (n_kernels, height - kernel_size + 1, width - kernel_size + 1)
        self.kernels_shape = (n_kernels, depth, kernel_size, kernel_size)
        
        # Xavier/Glorot initialization para los kernels
        limit = np.sqrt(6 / (depth * kernel_size * kernel_size + n_kernels))
        self.kernels = np.random.uniform(-limit, limit, self.kernels_shape)
        
        # Inicializar biases con la forma correcta
        self.biases = np.zeros(self.output_shape)

    def _im2col(self, input_data, kernel_size):
        """
        Transforma la entrada en una matriz que permite operaciones vectorizadas
        para la convolución
        """
        batch_size, depth, height, width = input_data.shape
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1
        
        # Inicializar la matriz de salida
        col = np.zeros((batch_size, depth, kernel_size, kernel_size, out_height, out_width))
        
        # Llenar la matriz con los valores correspondientes
        for y in range(kernel_size):
            y_max = y + out_height
            for x in range(kernel_size):
                x_max = x + out_width
                col[:, :, y, x, :, :] = input_data[:, :, y:y_max, x:x_max]
        
        # Reshape para obtener la matriz final
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        return col

    def _col2im(self, col, input_shape):
        """
        Función inversa de im2col para la propagación hacia atrás
        """
        batch_size, depth, height, width = input_shape
        kernel_size = self.kernel_size
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1
        
        # Inicializar la matriz de salida
        img = np.zeros((batch_size, depth, height, width))
        
        # Reshapear col para facilitar la operación inversa
        col_reshaped = col.reshape(batch_size, out_height, out_width, depth, kernel_size, kernel_size)
        col_reshaped = col_reshaped.transpose(0, 3, 4, 5, 1, 2)
        
        # Sumar los valores a la matriz de salida
        for y in range(kernel_size):
            y_max = y + out_height
            for x in range(kernel_size):
                x_max = x + out_width
                img[:, :, y:y_max, x:x_max] += col_reshaped[:, :, y, x, :, :]
        
        return img

    def forward(self, input):
        """
        Realiza la convolución utilizando operaciones matriciales
        """
        self.input = input
        batch_size, depth, height, width = input.shape
        
        # Crear la matriz de columnas
        self.col = self._im2col(input, self.kernel_size)
        
        # Reshape kernels para multiplicación matricial
        kernels_reshaped = self.kernels.reshape(self.n_kernels, -1)
        
        # Realizar la convolución como una multiplicación de matrices
        output = np.dot(self.col, kernels_reshaped.T)
        
        # Reshape para obtener el resultado final
        output = output.reshape(batch_size, height - self.kernel_size + 1, 
                                width - self.kernel_size + 1, self.n_kernels)
        output = output.transpose(0, 3, 1, 2)
        
        # Añadir biases (asegurándose de que la forma sea correcta para broadcasting)
        output += self.biases[np.newaxis, :, :, :]
        
        return output
    
    def backward(self, input_gradient, learning_rate):
        """
        Propaga el gradiente hacia atrás
        """
        batch_size = input_gradient.shape[0]
        
        # Reshape input_gradient para multiplicación matricial
        input_gradient_reshaped = input_gradient.transpose(0, 2, 3, 1).reshape(-1, self.n_kernels)
        
        # Calcular el gradiente de los kernels
        kernels_gradient = np.dot(input_gradient_reshaped.T, self.col)
        kernels_gradient = kernels_gradient.reshape(self.kernels_shape)
        
        # Calcular el gradiente de la entrada
        col_gradient = np.dot(input_gradient_reshaped, self.kernels.reshape(self.n_kernels, -1))
        output_gradient = self._col2im(col_gradient, self.input.shape)
        
        # Calcular el gradiente del bias y asegurarse de que tenga la forma correcta
        # Sumar sobre el batch (eje 0) y mantener las dimensiones espaciales
        biases_gradient = np.sum(input_gradient, axis=0)
        
        # Actualizar parámetros
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * biases_gradient
        
        return output_gradient