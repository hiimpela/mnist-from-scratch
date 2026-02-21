import pickle
import gzip
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(base_path, "data", "mnist.pkl.gz")

def load_data(path = default_path):
    with gzip.open(path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper(path = default_path, expand = False):
    tr_d, va_d, te_d = load_data(path)

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    if expand:
        expanded_training_data = []
        for x, y in training_data:
            expanded_training_data.append((x, y))
            image = x.reshape(28, 28)

            for d, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
                shifted_image = np.roll(image, d, axis)
                expanded_training_data.append((shifted_image.reshape(784, 1), y))
        training_data = expanded_training_data

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


