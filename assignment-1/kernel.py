import numpy as np
import math


def softmax(A):
    max_value = np.max(A,axis=0)
    exp = np.exp(A - max_value)
    exp_sum = np.sum(exp,axis=0)
    dist = exp / exp_sum
    return dist, A - max_value


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def tanh(x):
    y1 = (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))
    cache = x
    return y1, cache


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / float(m)
    db = np.sum(dZ, axis=1, keepdims=True) / float(m)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def tanh_backward(dA, cache):
    Z = cache
    s1 = np.exp(Z) - np.exp(-Z)
    s2 = np.exp(Z) + np.exp(-Z)
    tanh = s1 / s2
    s = dA * (1 - tanh * tanh)
    return s


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def softmax_backward(Y, cache):
    Z = cache  # cache = A - max_value
    s = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    dZ = s - Y

    return dZ

def linear_activation_forward(A_prev, W, b, kernel):
    if kernel == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif kernel == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif kernel == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif kernel == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def linear_activation_backward(dA, cache, kernel):
    linear_cache, activation_cache = cache

    if kernel == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif kernel == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif kernel =="tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif kernel == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters