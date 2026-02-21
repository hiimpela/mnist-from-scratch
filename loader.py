import network2
import mnist_loader

# 1. Load the data to have something to test against
_, _, test_data = mnist_loader.load_data_wrapper()

# 2. Call your load function with the specific filename
net = network2.load("trained_model.json")

# 3. Check the accuracy to ensure it matches your 99% training result
print(f"Checking accuracy of {net.sizes} model...")
accuracy = net.accuracy(test_data)
print(f"Final Test Accuracy: {accuracy} / 10000")