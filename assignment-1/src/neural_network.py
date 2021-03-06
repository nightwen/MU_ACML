# coding=utf-8
import numpy as np
import kernel
import matplotlib.pyplot as plt


def generate_dataset(size):
    """
    Returns input and output data.
    """
    X = np.identity(size)
    return X, X.copy()


def initialize_parameters(layers_shape, seed=23):
    np.random.seed(seed)
    parameters = {}
    L = len(layers_shape)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_shape[l],
                                                   layers_shape[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_shape[l], 1))
    return parameters


def compute_cost(AL, Y):  # this works for binary loss, not valid for our homework
    m = Y.shape[1]
    test1 = np.multiply(Y, np.log(AL))
    cost = - np.sum(np.multiply(Y, np.log(AL)) +
                    np.multiply(1 - Y, np.log(1 - AL))) / m
    cost = np.squeeze(cost)
    return cost


def multiple_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.sum(Y * np.log(AL))) / float(m)
    return cost


def binary_cost(AL, Y):
    cost = - np.sum(np.multiply(Y, np.log(AL)) +
                    np.multiply(1 - Y, np.log(1 - AL))) / Y.shape[1]
    return cost


def train_model(X, Y, layers_shape, learning_rate=0.075, iterations=5000, log_costs=True, save_graph=True):

    costs = []
    parameters = initialize_parameters(layers_shape, seed=1)
    L = len(parameters) // 2

    for i in range(0, iterations):

        # these two should be init in each loop
        L_cache = []  # the set for the data of forward
        A = X

        # here all layer use a same kernel
        for l in range(1, L):
            A_prev = A
            A, cache = kernel.forward_propagation(
                A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'sigmoid')
            L_cache.append(cache)

        A, cache = kernel.forward_propagation(
            A, parameters['W' + str(L)], parameters['b' + str(L)], 'softmax')
        L_cache.append(cache)
        grads = {}  # record the each gradient of each layer
        L = len(L_cache)  # the number of layers
        current_cache = L_cache[L - 1]

        # binary classification
        cost = binary_cost(A, Y)
        # dAL = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        # grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = kernel.linear_activation_backward(dAL,
        #                                                                                                         current_cache,
        #                                                                                                         'sigmoid')

        # multiple classification
        cost = multiple_cost(A, Y)
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = kernel.backward_propagation(Y,
                                                                                                                 current_cache,
                                                                                                                 'softmax')

        for l in reversed(range(L-1)):
            current_cache = L_cache[l]
            dA_prev_temp, dW_temp, db_temp = kernel.backward_propagation(
                grads['dA' + str(l + 1)], current_cache, 'sigmoid')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        for l in range(L):
            parameters["W" + str(l + 1)] -= learning_rate * \
                grads['dW' + str(l + 1)]
            parameters["b" + str(l + 1)] -= learning_rate * \
                grads['db' + str(l + 1)]
        
        if log_costs:
            print("Cost after iteration %i: %f" % (i, cost))
        costs.append(cost)

    # plot cost
    if save_graph:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()
        plt.savefig("cost.png")
        plt.close()

    return parameters


def predict(X, parameters, threshold=0.5):
    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = kernel.forward_propagation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                                    'sigmoid')

    A, cache = kernel.forward_propagation(
        A, parameters['W' + str(L)], parameters['b' + str(L)], 'softmax')
    predict_label = np.where(A > 0.5, 1, 0)
    return A, predict_label


if __name__ == '__main__':
    X, Y = generate_dataset(8)
    layers_shape = (8, 3, 8)
    parameters = training_model(X, Y, layers_shape)
    possibility, predict_label = predict(X, parameters)
    print(predict_label)
