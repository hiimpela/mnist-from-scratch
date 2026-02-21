import numpy as np
import random
import json
import sys

class cross_entropy:
    @staticmethod
    def fn(a, y):
        a = np.clip(a, 1e-15, 1.0 - 1e-15) # log(0) explodes so we clip
        return -np.sum(y * np.log(a))

    @staticmethod
    def delta(z, a, y):
        return (a - y)

class network():
    def __init__(self, sizes, cost = cross_entropy):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
    def default_weight_initializer(self):
        self.biases = [np.full((y, 1), 0.01) for y in self.sizes[1:]] # np full so the neurons arent dead at the start
        self.weights = [np.random.randn(y, x) / np.sqrt(x / 2.0) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):
            a = relu(np.dot(weight, a) + bias)
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1]) 
        return a
    
    def sgd(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0, evaluation_data = None, 
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False,
            learning_schedule = False):
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = [], [], [], []
        max_eta_drops = 7
        eta_drops = 0
        epochs_since_improvement = 0
        best_accuracy = 0

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert = True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert = True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_data))

            current_accuracy = self.accuracy(evaluation_data)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            
            if learning_schedule and epochs_since_improvement >= 10:
                eta /= 2
                eta_drops += 1
                epochs_since_improvement = 0
                print("Eta dropped to {}".format(eta))
                if eta_drops >= max_eta_drops:
                    print("Learning rate floor hit")
                    break
            max_w = max(np.max(np.abs(w)) for w in self.weights)
            print(f"Max weight magnitude: {max_w:.4f}")            
            print()
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        X = np.column_stack([data[0] for data in mini_batch])
        Y = np.column_stack([data[1] for data in mini_batch])

        nabla_b, nabla_w = self.backprop(X, Y)
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)
        
        #final layer softmax
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        error = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = np.sum(error, axis = 1, keepdims = True)
        nabla_w[-1] = np.dot(error, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            error = np.dot(self.weights[-l + 1].transpose(), error) * sp
            nabla_b[-l] = np.sum(error, axis = 1, keepdims = True)
            nabla_w[-l] = np.dot(error, activations[-l -1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert = False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert = False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorize(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        data = {"sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": self.cost.__name__}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def softmax(z):
    exp = np.exp(z - np.max(z, axis = 0, keepdims = True))
    return exp / np.sum(exp, axis = 0, keepdims = True)

def vectorize(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e
