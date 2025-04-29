import numpy as np

"""
    Todas las funciones de pérdida reciben:
        y_true (numpy.ndarray): Valores reales.
        y_pred (numpy.ndarray): Predicciones del modelo.
"""

def mse(y_true, y_pred):
    """Calcula el error cuadrático medio."""
    return np.mean((y_true - y_pred)**2)
    
def mse_prime(y_true, y_pred):
    """Derivada del MSE respecto a y_pred."""
    return 2 * (y_pred - y_true) / np.size(y_true)

def accuracy(y_true, y_pred):
    """Calcula la precisión (porcentaje de aciertos)."""
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    return correct / total

def categorical_cross_entropy(y_true, y_pred):
    """Calcula la pérdida de entropía cruzada categórica."""
    # Evitar log(0) agregando un pequeño valor epsilon
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calcular la pérdida
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]  # Promedio sobre el batch
    return loss


def categorical_cross_entropy_prime(y_true, y_pred):
    """Calcula el gradiente de la pérdida de entropía cruzada categórica con respecto a y_pred."""
    # El gradiente es (y_pred - y_true) / batch_size
    batch_size = y_pred.shape[0]
    return (y_pred - y_true) / batch_size