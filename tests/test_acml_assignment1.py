from acml_assignment1 import __version__
from acml_assignment1 import NerualNetwork
import numpy as np

def test_version():
    assert __version__ == '0.1.0'


def test_predict():
    X, Y = NerualNetwork.generate_dataset(8)
    layers_shape = (8, 3, 8)
    parameters = NerualNetwork.training_model(X, Y, layers_shape)
    X_test = np.array([[0, 0, 1, 0, 0, 0, 0, 0]]).T
    possibility, predict_label = NerualNetwork.get_prediction(X_test, parameters)
    print (predict_label)


if __name__ == '__main__':
    test_predict()