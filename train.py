import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(expand = True)
import ml.mnist.network as network
net = network.network([784, 800, 10], cost=network.cross_entropy)
net.sgd(training_data = training_data, 
        epochs = 0, 
        mini_batch_size = 128, 
        eta = 0.05, #learning rate 
        lmbda = 5.0, 
        evaluation_data=validation_data, 
        monitor_evaluation_accuracy=True, 
        monitor_evaluation_cost=True, 
        monitor_training_accuracy=False, 
        monitor_training_cost=False, 
        learning_schedule=True)
print("Training done! Now saving")
net.save("trained_model.json")
print("Model saved successfully as 'trained_model.json'!")