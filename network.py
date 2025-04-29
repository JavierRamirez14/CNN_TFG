import numpy as np
import time
from dense import Dense
from activations import ReLU, Softmax
from loss import mse, mse_prime, accuracy

def train_batch(X, y, net, loss, loss_prime, learning_rate):
    """
    Entrena un batch completo de datos y devuelve el error promedio.
    
    Args:
        X (numpy.ndarray): Datos de entrada.
        y (numpy.ndarray): Etiquetas verdaderas.
        net (list): Lista de capas de la red.
        loss (function): Función de pérdida.
        loss_prime (function): Derivada de la función de pérdida.
        learning_rate (float): Tasa de aprendizaje.
    """
    # Forward pass
    input = X
    for layer in net:
        input = layer.forward(input)
    y_pred = input
    
    # Cálculo de error
    error = loss(y, y_pred)
    
    # Backward pass
    grad = loss_prime(y, y_pred)
    for layer in reversed(net):
        grad = layer.backward(grad, learning_rate)
    
    # Devolver error, predicciones y etiquetas verdaderas
    return error, y_pred, y

def train(data, net, loss, loss_prime, epochs, learning_rate):
    """
    Entrena la red durante varias épocas.
    
    Args:
        data (iterable): Iterable que proporciona batches de datos (X, y).
        net (list): Lista de capas que forman la red neuronal.
        loss (function): Función de pérdida a utilizar (ej: mse).
        loss_prime (function): Derivada de la función de pérdida.
        epochs (int): Número de épocas de entrenamiento.
        learning_rate (float): Tasa de aprendizaje para la actualización de pesos.
    """
def train(train_data, val_data, net, loss, loss_prime, epochs, learning_rate):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_error = 0
        batch_count = 0
        
        # Almacenar predicciones y etiquetas verdaderas de entrenamiento
        for X, y in train_data:
            y_train_preds = np.empty((0, y.shape[1]))  
            y_train_true = np.empty((0, y.shape[1]))
            break
        
        for X, y in train_data:
            # Usar la función train_batch mejorada
            error, batch_preds, batch_true = train_batch(X, y, net, loss, loss_prime, learning_rate)
            
            # Acumular predicciones y etiquetas para calcular accuracy
            y_train_preds = np.concatenate((y_train_preds, batch_preds), axis=0)
            y_train_true = np.concatenate((y_train_true, batch_true), axis=0)
            
            batch_count += 1
            epoch_error += error
        
        # Calcular pérdida promedio en entrenamiento
        epoch_error /= batch_count
        
        # Calcular accuracy en entrenamiento
        y_train_pred_class = np.argmax(y_train_preds, axis=1)
        y_train_true_class = np.argmax(y_train_true, axis=1)
        train_acc = accuracy(y_train_true_class, y_train_pred_class)
        
        # Calcular métricas en validación
        val_error, val_acc = evaluate(val_data, net, loss)
        
        # Mostrar métricas de la época
        print(f'\rEpoch: {epoch+1}/{epochs} | Train Loss: {epoch_error:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_error:.4f} | Val Acc: {val_acc:.4f}')
        
        # Guardar métricas en el historial
        history['train_loss'].append(epoch_error)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_error)
        history['val_acc'].append(val_acc)
        
        end_time = time.time()
        print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")
    
    return history

def evaluate(data, net, loss):
    
    for X, y in data:
            y_preds = np.empty((0, y.shape[1]))  
            y_true = np.empty((0, y.shape[1]))
            break
    
    for X, y in data:
        # Forward
        input = X
        for layer in net:
            input = layer.forward(input)
        y_pred = input
        y_preds = np.concatenate((y_preds, y_pred), axis=0)
        y_true = np.concatenate((y_true, y), axis=0)

    # Calcular loss
    error = loss(y_true, y_preds)

    # Calcular accuracy
    y_pred_class = np.argmax(y_preds, axis=1)
    y_true_class = np.argmax(y_true, axis=1)
    acc = accuracy(y_true_class, y_pred_class)

    return error, acc


def test(X, y, net):
    """Evalúa la red y devuelve el accuracy."""
    # Forward pass
    input = X
    for layer in net:
        input = layer.forward(input)
    
    # Calcula predicciones y accuracy
    y_pred = np.argmax(input, axis=1)
    y_true = np.argmax(y, axis=1)
    return accuracy(y_true, y_pred)