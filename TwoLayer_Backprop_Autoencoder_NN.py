import numpy as np
import matplotlib.pyplot as plt


class TwoLayerBackpropAutoencoderNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """
        Initialize the two-layer backpropagation autoencoder neural network.

        Args:
            input_size (int): The number of input neurons.
            hidden_size1 (int): The number of neurons in the first hidden layer.
            hidden_size2 (int): The number of neurons in the second hidden layer.
            output_size (int): The number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size1)
        self.b1 = np.zeros(self.hidden_size1)
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.b2 = np.zeros(self.hidden_size2)
        self.W3 = np.random.randn(self.hidden_size2, self.output_size)
        self.b3 = np.zeros(self.output_size)

    def forward(self, X):
        """
        Perform the forward pass of the neural network.

        Args:
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output of the neural network.
        """
        # Calculate the first layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Calculate the second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        # Calculate the third layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    def sigmoid(self, z):
        """
        Calculate the sigmoid activation function.

        Args:
            z (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output of the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """
        Calculate the derivative of the sigmoid activation function.

        Args:
            z (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output of the derivative of the sigmoid activation function.
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def backward(self, X, y, output):
        """
        Perform the backward pass of the neural network.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            output (numpy.ndarray): The output of the neural network.
        """
        # Calculate the error and delta of the output layer
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(self.z3)

        # Calculate the error and delta of the second hidden layer
        self.a2_error = self.output_delta.dot(self.W3.T)
        self.a2_delta = self.a2_error * self.sigmoid_derivative(self.z2)

        # Calculate the error and delta of the first hidden layer
        self.a1_error = self.a2_delta.dot(self.W2.T)
        self.a1_delta = self.a1_error * self.sigmoid_derivative(self.z1)

        # Update the weights and biases
        self.W1 += X.T.dot(self.a1_delta)
        self.b1 += np.sum(self.a1_delta, axis=0)
        self.W2 += self.a1.T.dot(self.a2_delta)
        self.b2 += np.sum(self.a2_delta, axis=0)
        self.W3 += self.a2.T.dot(self.output_delta)
        self.b3 += np.sum(self.output_delta, axis=0)

    def train(self, X, y, epochs):
        """
        Train the neural network.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            epochs (int): The number of epochs to train the neural network.
        """
        # Initialize the loss
        self.loss = []

        # Train the neural network
        for i in range(epochs):
            # Perform the forward pass
            output = self.forward(X)

            # Perform the backward pass
            self.backward(X, y, output)

            # Calculate the loss
            self.loss.append(np.mean(np.square(y - output)))

    def predict(self, X):
        """
        Predict the output of the neural network.

        Args:
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output of the neural network.
        """
        return self.forward(X)


if __name__ == '__main__':
    # Load the dataset
    X = np.load('mnist_train_images.npy')
    y = np.load('mnist_train_labels.npy')

    # Normalize the input
    X = X / 255.0

    # Initialize the model
    model = TwoLayerBackpropAutoencoderNN(input_size=784, hidden_size1=128, hidden_size2=64, output_size=784)

    # Train the model
    model.train(X, X, epochs=100)

    # Plot the loss
    plt.plot(model.loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()