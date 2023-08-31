import numpy as np
import matplotlib.pyplot as plt


class OneLayerBackpropAutoencoderNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the one-layer backpropagation autoencoder neural network.

        Args:
            input_size (int): The number of input neurons.
            hidden_size (int): The number of hidden neurons.
            output_size (int): The number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros(self.output_size)

    def forward(self, X):
        """
        Perform the forward pass of the neural network.

        Args:
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output of the neural network.
        """
        # Calculate the first layer
        self.z = np.dot(X, self.W1) + self.b1
        self.a = self.sigmoid(self.z)

        # Calculate the second layer
        self.z2 = np.dot(self.a, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

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
        self.output_delta = self.output_error * self.sigmoid_derivative(self.z2)

        # Calculate the error and delta of the hidden layer
        self.a_error = self.output_delta.dot(self.W2.T)
        self.a_delta = self.a_error * self.sigmoid_derivative(self.z)

        # Update the weights and biases
        self.W1 += X.T.dot(self.a_delta)
        self.b1 += np.sum(self.a_delta, axis=0)
        self.W2 += self.a.T.dot(self.output_delta)
        self.b2 += np.sum(self.output_delta, axis=0)

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
    model = OneLayerBackpropAutoencoderNN(input_size=784, hidden_size=64, output_size=784)

    # Train the model
    model.train(X, X, epochs=100)

    # Plot the loss
    plt.plot(model.loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
