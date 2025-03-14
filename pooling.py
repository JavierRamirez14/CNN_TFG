import numpy as np
from capa import Capa

class MaxPooling(Capa):
    def __init__(self, input_shape, pool_size=2, stride=2):
        """
        Capa de Max Pooling
        
        Parámetros:
            input_shape: Tupla (depth, height, width)
            pool_size: Tamaño de la ventana de pooling (cuadrada)
            stride: Paso de la ventana de pooling
        """
        self.input_shape = input_shape
        self.depth, self.height, self.width = input_shape
        self.pool_size = pool_size
        self.stride = stride
        
        # Calcular la forma de la salida
        output_height = (self.height - pool_size) // stride + 1
        output_width = (self.width - pool_size) // stride + 1
        self.output_shape = (self.depth, output_height, output_width)
        
    def forward(self, input):
        """
        Propagación hacia adelante
        """
        self.input = input
        batch_size, depth, height, width = input.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        # Inicializar la salida
        output = np.zeros((batch_size, depth, output_height, output_width))
        
        # Guardar las posiciones de los valores máximos para el backward
        self.max_indices = np.zeros((batch_size, depth, output_height, output_width, 2), dtype=int)
        
        # Realizar el pooling
        for b in range(batch_size):
            for d in range(depth):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Extraer la región de la ventana
                        window = input[b, d, h_start:h_end, w_start:w_end]
                        
                        # Encontrar el valor máximo y su posición
                        max_val = np.max(window)
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        
                        # Guardar el valor máximo y su posición
                        output[b, d, h, w] = max_val
                        self.max_indices[b, d, h, w] = max_pos
        
        return output
    
    def backward(self, input_gradient, learning_rate):
        """
        Propagación hacia atrás
        """
        batch_size, depth, output_height, output_width = input_gradient.shape
        output_gradient = np.zeros_like(self.input)
        
        # Distribuir el gradiente a las posiciones donde estaban los valores máximos
        for b in range(batch_size):
            for d in range(depth):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # Obtener las coordenadas del valor máximo en la ventana
                        max_h, max_w = self.max_indices[b, d, h, w]
                        
                        # Asignar el gradiente
                        output_gradient[b, d, h_start + max_h, w_start + max_w] += input_gradient[b, d, h, w]
        
        return output_gradient


class AveragePooling(Capa):
    def __init__(self, input_shape, pool_size=2, stride=2):
        """
        Capa de Average Pooling
        
        Parámetros:
            input_shape: Tupla (depth, height, width)
            pool_size: Tamaño de la ventana de pooling (cuadrada)
            stride: Paso de la ventana de pooling
        """
        self.input_shape = input_shape
        self.depth, self.height, self.width = input_shape
        self.pool_size = pool_size
        self.stride = stride
        
        # Calcular la forma de la salida
        output_height = (self.height - pool_size) // stride + 1
        output_width = (self.width - pool_size) // stride + 1
        self.output_shape = (self.depth, output_height, output_width)
        
    def forward(self, input):
        """
        Propagación hacia adelante
        """
        self.input = input
        batch_size, depth, height, width = input.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        # Inicializar la salida
        output = np.zeros((batch_size, depth, output_height, output_width))
        
        # Realizar el pooling
        for b in range(batch_size):
            for d in range(depth):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Extraer la región de la ventana
                        window = input[b, d, h_start:h_end, w_start:w_end]
                        
                        # Calcular el promedio
                        output[b, d, h, w] = np.mean(window)
        
        return output
    
    def backward(self, input_gradient, learning_rate):
        """
        Propagación hacia atrás
        """
        batch_size, depth, output_height, output_width = input_gradient.shape
        output_gradient = np.zeros_like(self.input)
        
        # Distribuir el gradiente uniformemente en cada ventana
        for b in range(batch_size):
            for d in range(depth):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Distribuir el gradiente uniformemente
                        output_gradient[b, d, h_start:h_end, w_start:w_end] += \
                            input_gradient[b, d, h, w] / (self.pool_size * self.pool_size)
        
        return output_gradient


# Versión optimizada de MaxPooling usando im2col para mayor velocidad
class FastMaxPooling(Capa):
    def __init__(self, input_shape, pool_size=2, stride=2):
        """
        Implementación más rápida de Max Pooling usando técnica similar a im2col
        """
        self.input_shape = input_shape
        self.depth, self.height, self.width = input_shape
        self.pool_size = pool_size
        self.stride = stride
        
        output_height = (self.height - pool_size) // stride + 1
        output_width = (self.width - pool_size) // stride + 1
        self.output_shape = (self.depth, output_height, output_width)
    
    def _im2col_indices(self, input_shape, pool_size, stride):
        """
        Calcula los índices para la transformación im2col
        """
        batch_size, depth, height, width = input_shape
        output_height = (height - pool_size) // stride + 1
        output_width = (width - pool_size) // stride + 1
        
        # Generar índices para cada dimensión
        i0 = np.repeat(np.arange(pool_size), pool_size)
        i1 = np.repeat(np.arange(output_height) * stride, output_width)
        j0 = np.tile(np.arange(pool_size), pool_size)
        j1 = np.tile(np.arange(output_width) * stride, output_height)
        
        # Combinar índices
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        
        return i, j
    
    def forward(self, input):
        """
        Propagación hacia adelante usando técnica de im2col
        """
        self.input = input
        batch_size, depth, height, width = input.shape
        
        # Generar índices
        i, j = self._im2col_indices(input.shape, self.pool_size, self.stride)
        
        # Crear columnas
        cols = np.zeros((batch_size, depth, self.pool_size**2, i.shape[1]))
        
        # Llenar columnas con valores de input
        for b in range(batch_size):
            for d in range(depth):
                cols[b, d] = input[b, d, i, j]
        
        # Encontrar valores máximos
        max_idx = np.argmax(cols, axis=2)
        self.max_idx = max_idx
        
        # Crear output con reshape adecuado
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, depth, output_height, output_width))
        
        # Llenar output con valores máximos
        for b in range(batch_size):
            for d in range(depth):
                output[b, d] = np.choose(max_idx[b, d], cols[b, d]).reshape(output_height, output_width)
        
        return output
    
    def backward(self, input_gradient, learning_rate):
        """
        Propagación hacia atrás usando índices guardados
        """
        batch_size, depth, output_height, output_width = input_gradient.shape
        output_gradient = np.zeros_like(self.input)
        
        # Generar índices
        i, j = self._im2col_indices(self.input.shape, self.pool_size, self.stride)
        
        # Distribuir gradientes
        for b in range(batch_size):
            for d in range(depth):
                # Aplanar gradiente de entrada
                flat_grad = input_gradient[b, d].flatten()
                
                # Crear matriz de gradientes para cada posición
                grads = np.zeros((self.pool_size**2, flat_grad.shape[0]))
                
                # Marcar las posiciones de los máximos
                max_idx = self.max_idx[b, d]
                for idx, max_pos in enumerate(max_idx):
                    grads[max_pos, idx] = flat_grad[idx]
                
                # Sumar contribuciones para cada posición
                for k in range(self.pool_size**2):
                    output_gradient[b, d, i[k], j[k]] += grads[k]
        
        return output_gradient