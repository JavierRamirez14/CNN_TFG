import numpy as np
from dense import Dense
from activations import ReLU, Softmax
from loss import mse, mse_prime, accuracy

def train_batch(X, y, net, loss, loss_prime, learning_rate):
    # Forward
    input = X
    for layer in net:
        input = layer.forward(input)
    y_pred = input

    # Error
    error = loss(y, y_pred)

    # Backward
    grad = loss_prime(y, y_pred)
    for layer in reversed(net):
        grad = layer.backward(grad, learning_rate)

    return error

def train(data, net, loss, loss_prime, epochs, learning_rate):
    for epoch in range(epochs):
        epoch_error = 0
        i = 0
        for X, y in data:
            error = train_batch(X, y, net, loss, loss_prime, learning_rate)
            i += 1

            epoch_error += error

        epoch_error /= i
        print(f'Epoch: {epoch+1} | Error: {epoch_error}')

def test(X, y, net):
    # Forward pass
    input = X
    for layer in net:
        input = layer.forward(input)
    
    y_pred = np.array([np.argmax(x) for x in input])
    y_true = np.array([np.argmax(x) for x in y])

    # Calcular el accuracy
    return accuracy(y_true, y_pred)
