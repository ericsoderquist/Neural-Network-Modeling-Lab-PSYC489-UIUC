# Hopfield Network: An Advanced Framework for Associative Memory
### University of Illinois Urbana-Champaign
#### Author: Eric Soderquist
#### Project Repository: [GitHub](https://github.com/ericsoderquist/Neural-Network-Modeling-Lab-PSYC489-UIUC/tree/main/Assignments/Hopfield-Network)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Design Decisions](#design-decisions)
4. [Prerequisites](#prerequisites)
5. [File Structure](#file-structure)
6. [Implementation Overview](#implementation-overview)
7. [Usage](#usage)
8. [Performance Metrics](#performance-metrics)
9. [Contributions](#contributions)
10. [Acknowledgments](#acknowledgments)
11. [License](#license)

---

### Introduction

The Hopfield Network project aims to implement an advanced framework for associative memory using a Hopfield network model. The model serves as a powerful tool for the exploration and understanding of memory retrieval in both artificial and biological neural networks. This repository houses the Python codebase and relevant documentation, providing a comprehensive resource for scholars, engineers, and anyone interested in neural networks and associative memory.

### Theoretical Background

The Hopfield Network is a recurrent neural network architecture named after John Hopfield, who introduced it in 1982. It operates under the principles of symmetrically weighted connections and energy minimization. Unlike feedforward neural networks, the Hopfield network features recurrent connections, enabling it to function as an associative memory system. This repository's implementation adheres to these foundational principles, incorporating a robust mathematical model to simulate the network's dynamics accurately.

### Design Decisions

The Python implementation uses a class-based structure to encapsulate the functionality of the Hopfield Network. The key attributes and methods in the class are as follows:

- `n_units`: Represents the number of neurons in the network.
- `threshold`: The activation threshold.
- `weights`: A NumPy ndarray storing the synaptic weights.
- `train(patterns)`: A method to train the network with a set of patterns.
- `update(pattern, async_update=True)`: Updates the network state, allowing for both synchronous and asynchronous updates.
- `energy(pattern)`: Calculates the energy of the current network state.

The implementation allows for modular extension and easy integration with other machine learning frameworks.

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib (for visualization)

### File Structure

- `Hopfield-Network.py`: Python script containing the Hopfield Network class implementation.
- `README.md`: This document.
- `data/`: Directory containing sample patterns for training and testing.

### Implementation Overview

The `HopfieldNetwork` class serves as the primary interface for interacting with the network. The code is written in Python and adheres to the PEP 8 style guide. The mathematical model underpinning the network adheres to the original equations proposed by John Hopfield, ensuring accurate simulations.

### Usage

Please refer to the [Hopfield-Network.py](./Assignments/Hopfield-Network/Hopfield-Network.py) script for example usage. Training and state update methods are included in the class definition, allowing for straightforward use in research or applications.

### Performance Metrics

The performance of the Hopfield Network can be evaluated using metrics such as recall accuracy, stability, and energy landscape analysis. Further metrics and evaluation techniques are planned for future releases.

### Contributions

Contributions are welcome! Please read the contributing guidelines and code of conduct before submitting a pull request.

### Acknowledgments

Special thanks to the University of Illinois Urbana-Champaign and the Brain and Cognitive Sciences Department for their support and resources.

### License

MIT License. See [LICENSE](LICENSE) file for details.

---

