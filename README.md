
# MNIST from Scratch

A fully connected neural network built from scratch using `numpy` to classify handwritten digits. This project demonstrates the core mathematics of deep learning, avoiding heavy ML frameworks, and includes a real-time `pygame` canvas to test the model with your own handwriting.

## Features

* **Custom Neural Network Engine (`network.py`):** Implements forward pass, backpropagation, and Stochastic Gradient Descent (SGD) via raw matrix operations.
* **Architecture & Optimization:**
  * Hidden layers use ReLU activations, paired with a Softmax output layer.
  * He initialization combined with a 0.01 bias initialization prevents dead ReLU neurons.
  * Uses Cross-Entropy cost, L2 Regularization (weight decay), and a dynamic learning rate schedule that halves $\eta$ when evaluation accuracy plateaus.
* **Interactive Demo (`demo.py`):** A drawing interface that uses `scipy.ndimage` to calculate the center of mass of your drawing and translates it to the exact center of a 28x28 grid, matching the strict preprocessing of the original MNIST dataset.

## Project Structure

* `network.py`: The mathematical engine containing the network class, activation functions, and training logic.
* `train.py`: The execution script to load data, initialize the network, run SGD, and save the weights and biases.
* `demo.py`: The PyGame inference UI that loads a trained model and predicts drawn digits in real-time.
* `mnist_loader.py` & `loader.py`: Utilities to parse and format the raw MNIST dataset.
* `trained_model.json`: The serialized weights, biases, and architecture sizes of the pre-trained network.

## Requirements

Ensure Python 3 is installed along with the required dependencies:

```bash
pip install numpy scipy pygame
```

## Usage

### Running the Interactive Demo
You can launch the demo immediately using the pre-trained model file included in the repository. 

```bash
python demo.py
```
* **Draw:** Click and drag inside the main canvas.
* **Clear:** Click the red "Clear" button to reset the canvas.
* **Prediction:** The bottom left displays the 28x28 normalized image fed into the network, and the calculated prediction is displayed on the right.

### Training the Model
To train the model from scratch, execute the training script. 

*Note: Edit `train.py` to change `epochs = 0` to a positive integer (e.g., `epochs = 30`) before running to perform training iterations.*

```bash
python train.py
```
This configures a network with the architecture `[784, 800, 10]`, a learning rate ($\eta$) of 0.05, and L2 regularization ($\lambda$) of 5.0. The trained model state is saved to `trained_model.json` upon completion.
