import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
    
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    print(correct, total)
    return correct / total

import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    """
    Calcula la pérdida de entropía cruzada categórica.
    
    Parámetros:
        y_true: Etiquetas verdaderas (one-hot encoded), de forma (batch_size, num_classes).
        y_pred: Predicciones de la red (salida de softmax), de forma (batch_size, num_classes).
    
    Retorna:
        Pérdida de entropía cruzada categórica.
    """
    # Evitar log(0) agregando un pequeño valor epsilon
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calcular la pérdida
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]  # Promedio sobre el batch
    return loss


def categorical_cross_entropy_prime(y_true, y_pred):
    """
    Calcula el gradiente de la pérdida de entropía cruzada categórica con respecto a y_pred.
    
    Parámetros:
        y_true: Etiquetas verdaderas (one-hot encoded), de forma (batch_size, num_classes).
        y_pred: Predicciones de la red (salida de softmax), de forma (batch_size, num_classes).
    
    Retorna:
        Gradiente de la pérdida con respecto a y_pred.
    """
    # El gradiente es (y_pred - y_true) / batch_size
    batch_size = y_pred.shape[0]
    return (y_pred - y_true) / batch_size