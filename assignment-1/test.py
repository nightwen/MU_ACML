import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import neural_network
import kernel


def hyperparameter_search():
    X, Y = neural_network.generate_dataset(8)
    layers_shape = (8, 3, 8)
    learning_performance = []
    for learning_rate in [0, 0.03, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
        for iterations in [0, 3, 10, 30, 100, 300, 1000, 3000, 10000]:
            print("learning rate", learning_rate, "iterations", iterations)
            # `parameters` are the weights and biases
            parameters = neural_network.train_model(
                X, Y, layers_shape,
                learning_rate=learning_rate, iterations=iterations,
                log_costs=False, save_graph=False)

            numCorrect = 0
            for x in X:
                x = np.array([x]).T
                h, predicted_label = neural_network.predict(x, parameters)
                if np.array_equiv(x, predicted_label):
                    numCorrect += 1
            learning_performance.append(
                [learning_rate, iterations, numCorrect / x.size])

    # create diagram

    frame = pd.DataFrame(learning_performance, columns=[
                         "Learning rate", "#Iterations", "Accuracy"])
    wide_frame = pd.pivot_table(
        frame,
        values="Accuracy",
        index=["Learning rate"],
        columns="#Iterations"
    )
    sns.heatmap(
        wide_frame,
        cmap="viridis",
        annot=True
    )

    plt.savefig("learning_performance.png")
    plt.close()


def analysis():
    X, Y = neural_network.generate_dataset(8)
    layers_shape = (8, 3, 8)
    parameters = neural_network.train_model(
        X, Y, layers_shape,
        learning_rate=10, iterations=100,
        log_costs=False, save_graph=True)

    hidden, _ = kernel.sigmoid(parameters['W1'] + parameters['b1'])
    z = list(zip(*hidden))
    # permutation for creating the truth-table-like order
    ordered = [z[1], z[7], z[2], z[5], z[0], z[3], z[6], z[4]]
    # permutation for n = 16:
    # [z[0], z[6], z[8], z[2], z[14], z[12], z[1], z[4],
    #         z[3], z[15], z[9], z[5], z[11], z[7], z[13],  z[10]]

    frame = pd.DataFrame(ordered, columns=[
                         "Unit 1", "Unit 2", "Unit 3"])
    sns.heatmap(
        frame,
        cmap="viridis",
        square=True,
        annot=True
    )

    plt.savefig("hidden1.png")
    plt.close()

    hidden, _ = kernel.sigmoid(parameters['W2'].T + parameters['b2'].T)
    z = list(zip(*hidden))
    # permutation for creating the truth-table-like order
    ordered = [z[1], z[7], z[2], z[5], z[0], z[3], z[6], z[4]]

    frame = pd.DataFrame(ordered, columns=[
                         "Unit 1", "Unit 2", "Unit 3"])
    sns.heatmap(
        frame,
        cmap="viridis",
        square=True,
        annot=True
    )

    plt.savefig("hidden2.png")
    plt.close()


if __name__ == '__main__':
    hyperparameter_search()
    analysis()
