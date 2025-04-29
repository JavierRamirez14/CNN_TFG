# Estructura general de una capa
class Capa:
    def __init__(self):
        self.input = None # Almacena la entrada de la capa
        self.output = None # Almacena la salida de la capa

    def forward(self, input):
        # Propagación hacia adelante
        pass

    def backward(self, output_gradient, learning_rate):
        # Retropropagación
        pass