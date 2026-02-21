import network as network
import mnist_loader

_, _, test_data = mnist_loader.load_data_wrapper()

net = network.load("trained_model.json")

print(f"Checking accuracy of {net.sizes} model...")
accuracy = net.accuracy(test_data)
print(f"Final Test Accuracy: {accuracy} / 10000")