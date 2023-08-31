# Backpropagation-Autoencoders: A Comprehensive Investigation for Neural Network Modeling Lab PSYC 489
## University of Illinois Urbana-Champaign
### Author: Eric Soderquist
#### Founder & Lead Machine Learning Engineer / Data Scientist at Speakease

---

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Prerequisites](#prerequisites)
5. [File Structure](#file-structure)
6. [Methods Overview](#methods-overview)
7. [Usage](#usage)
8. [Theoretical Background](#theoretical-background)
9. [Performance Metrics](#performance-metrics)
10. [License](#license)

---

## Introduction

This repository showcases an individual assignment by Eric Soderquist, aiming to serve as a portfolio booster for job search in the fields of Machine Learning and Data Science. It contains a meticulous exploration and implementation of autoencoders utilizing backpropagation algorithms. Two autoencoder architectures—Single-layer and Two-layer autoencoders—are studied. The codebase is written in Python, leveraging its rich ecosystem of scientific computing and machine learning libraries.

---

## Motivation

Autoencoders have myriad applications in unsupervised learning, from dimensionality reduction to generative modeling. This repository aims to contribute to academic discourse, focusing on algorithmic efficiency, and scalability.

---

## Course Relevance

This project aligns with the objectives of PSYC 489, exploring key concepts of neural network modeling with a focus on autoencoders and backpropagation algorithms.

---

## Prerequisites

- Python 3.8+
- NumPy
- Scikit-learn
- Matplotlib

---

## File Structure

- `One_Layer_Backprop_Autoencoder_NN.py`: Implementation of a single-layer autoencoder using backpropagation.
    - Methods: `__init__`, `forward`, `sigmoid`, `sigmoid_derivative`, `backward`, `train`, `predict`
- `TwoLayer_Backprop_Autoencoder_NN.py`: Advanced two-layer autoencoder employing backpropagation.
    - Methods: `__init__`, `forward`, `sigmoid`, `sigmoid_derivative`, `backward`, `train`, `predict`

---

## Methods Overview

### Single-Layer Autoencoder (`One_Layer_Backprop_Autoencoder_NN.py`)

- `__init__`: Initializes the neural network architecture.
- `forward`: Conducts the forward pass.
- `sigmoid`: Implements the sigmoid activation function.
- `sigmoid_derivative`: Calculates the derivative of the sigmoid activation function.
- `backward`: Executes the backward pass.
- `train`: Trains the neural network.
- `predict`: Performs prediction based on the trained model.

### Two-Layer Autoencoder (`TwoLayer_Backprop_Autoencoder_NN.py`)

- `__init__`: Initializes the neural network architecture.
- `forward`: Conducts the forward pass.
- `sigmoid`: Implements the sigmoid activation function.
- `sigmoid_derivative`: Calculates the derivative of the sigmoid activation function.
- `backward`: Executes the backward pass.
- `train`: Trains the neural network.
- `predict`: Performs prediction based on the trained model.

---

## Usage

To execute the single-layer autoencoder:

```bash
python One_Layer_Backprop_Autoencoder_NN.py
```

To execute the two-layer autoencoder:

```bash
python TwoLayer_Backprop_Autoencoder_NN.py
```

---


## Theoretical Background

Autoencoders are a type of artificial neural network used for unsupervised learning. Their primary function is to encode the input data as internal fixed-size representations in reduced dimensionality and to reconstruct the output from this representation. The network architecture is symmetric, having an encoder function followed by a decoder function.

### Single-Layer Autoencoder

The single-layer autoencoder consists of three layers: an input layer, a hidden layer, and an output layer. The neurons in the hidden layer are activated by a nonlinear function, commonly the sigmoid function. The backpropagation algorithm is employed to minimize the reconstruction loss, typically the Mean Squared Error (MSE), between the input and the reconstructed output.

### Two-Layer Autoencoder

The two-layer autoencoder extends the single-layer architecture by adding an additional hidden layer. This allows the network to capture more complex features in the data. Similar to the single-layer autoencoder, the two-layer autoencoder also employs backpropagation for training but adapts it to optimize the increased complexity.

### Backpropagation Algorithm

Backpropagation is an optimization algorithm used for minimizing the error in predicting the output in comparison with the true or target output. It is a particular case of a more general optimization algorithm known as the gradient descent algorithm. The backpropagation algorithm works by computing the gradient of the loss function concerning each weight by the chain rule, updating the weights and biases to minimize the loss.



---

## Performance Metrics

- Reconstruction Loss
- Training Time
- Model Complexity

---

## Code Snippets

To give you a practical insight into how to interact with the code, here are some examples demonstrating the usage of key methods:

### Single-Layer Autoencoder

```python
# Initialization
autoencoder = OneLayerBackpropAutoencoderNN(input_size=784, hidden_size=128, output_size=784)

# Training
autoencoder.train(X_train, y_train, epochs=50)

# Prediction
reconstructed_data = autoencoder.predict(X_test)
```

### Two-Layer Autoencoder

```python
# Initialization
autoencoder = TwoLayerBackpropAutoencoderNN(input_size=784, hidden_size1=128, hidden_size2=64, output_size=784)

# Training
autoencoder.train(X_train, y_train, epochs=50)

# Prediction
reconstructed_data = autoencoder.predict(X_test)
```


## Contributions

I welcome contributions from students, researchers, and machine learning enthusiasts. If you find an area for improvement or wish to extend the functionalities, please feel free to make a pull request or contact me.



## License

MIT License. Please see the [LICENSE](LICENSE.md) file for more details.


