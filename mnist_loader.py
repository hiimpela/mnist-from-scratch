import pickle
import gzip
import numpy as np

def load_data(path="/home/maksiu/code/ml/neural-networks-and-deep-learning/data/mnist.pkl.gz"):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    with gzip.open(path, 'rb') as f:
        # Python 3 requires encoding when loading pickled data
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper(path="/home/maksiu/code/ml/neural-networks-and-deep-learning/data/mnist.pkl.gz"):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)`` in a format more convenient for neural networks."""
    tr_d, va_d, te_d = load_data(path)

    # Format training data (inputs are 784x1 vectors, outputs are one-hot vectors)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    # Validation data (inputs are reshaped, labels remain integers)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    # Test data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional one-hot vector with 1.0 at index j."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


