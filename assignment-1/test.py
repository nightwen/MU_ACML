import numpy as np
import neural_network


def test_version():
    assert __version__ == '0.1.0'


def test_predict():
    X, Y = neural_network.generate_dataset(8)
    layers_shape = (8, 3, 8)
    parameters = neural_network.training_model(X, Y, layers_shape)
    X_test = np.array([[0, 0, 1, 0, 0, 0, 0, 0]]).T
    possibility, predict_label = neural_network.get_prediction(
        X_test, parameters)
    print(predict_label)


if __name__ == '__main__':
    test_predict()
