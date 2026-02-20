import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2
net = network2.network([784, 100, 10], cost=network2.cross_entropy)
net.sgd(training_data, 600, 10, 1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, learning_schedule=True)
