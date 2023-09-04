"""
This module contains a HopfieldNetwork class that represents a Hopfield network. The class has the following attributes:
- n_units (int): the number of units in the network
- threshold (float): the threshold for activation
- weights (numpy.ndarray): the weights of the network

The class has the following methods:
- train(patterns): trains the network with the given patterns
- update(pattern, async_update=True): updates the network with the given pattern
- energy(pattern): calculates the energy of the given pattern

The module also contains the following functions:
- generate_test_patterns(patterns, probability): generates test patterns by flipping bits in the input patterns with a given probability.
- hamming_distance(pattern1, pattern2): calculates the hamming distance between two patterns.
- run_simulation(hopfield_network, patterns, test_patterns, max_iterations=100): runs a simulation of the Hopfield network with the given patterns and test patterns.
- generate_random_patterns(n_patterns, n_units): generates random patterns with the given number of patterns and units.
- generate_related_patterns(base_pattern, n_patterns, probability): generates related patterns based on a base pattern with the given number of patterns and probability.
"""
import numpy as np
import random

class HopfieldNetwork:
    """
    A class representing a Hopfield network.

    Attributes:
    - n_units (int): the number of units in the network
    - threshold (float): the threshold for activation
    - weights (numpy.ndarray): the weights of the network

    Methods:
    - train(patterns): trains the network with the given patterns
    - update(pattern, async_update=True): updates the network with the given pattern
    - energy(pattern): calculates the energy of the given pattern
    """
    def __init__(self, n_units, threshold=0):
        """
        Initializes a new instance of the HopfieldNetwork class.

        Args:
        - n_units (int): the number of units in the network
        - threshold (float): the threshold for activation
        """
        self.n_units = n_units
        self.threshold = threshold
        self.weights = np.zeros((n_units, n_units))

    def train(self, patterns):
        """
        Trains the network with the given patterns.

        Args:
        - patterns (list): a list of patterns to train the network with
        """
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def update(self, pattern, async_update=True):
        """
        Updates the network with the given pattern.

        Args:
        - pattern (list): the pattern to update the network with
        - async_update (bool): whether to use asynchronous or synchronous updating

        Returns:
        - the updated pattern as a list
        """
        pattern = np.array(pattern)
        if async_update:
            indices = np.random.permutation(self.n_units)
        else:
            indices = range(self.n_units)
        
        for index in indices:
            activation = np.dot(self.weights[index], pattern)
            pattern[index] = 1 if activation > self.threshold else 0
        return pattern.tolist()

    def energy(self, pattern):
        """
        Calculates the energy of the given pattern.

        Args:
        - pattern (list): the pattern to calculate the energy of

        Returns:
        - the energy of the pattern as a float
        """
        pattern = np.array(pattern)
        return -0.5 * np.dot(pattern.T, np.dot(self.weights, pattern))

patterns = [
    # Four binary patterns of 16 bits each.
    # Pattern 1: row of 8 "on" bits followed by 8 "off" bits.
    # Pattern 2: two rows of 4 "on" bits followed by 4 "off" bits.
    # Pattern 3: checkerboard pattern of alternating rows of 4 "on" and 4 "off" bits.
    # Pattern 4: alternating "on" and "off" bits in each column.
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
]

# Create a Hopfield network with 16 units and train it on a set of binary patterns.
hopfield_network = HopfieldNetwork(16)
hopfield_network.train(patterns)

def generate_test_patterns(patterns, probability):
    """
    Generate test patterns by flipping bits in the input patterns with a given probability.

    Args:
        patterns (list): A list of binary patterns.
        probability (float): The probability of flipping a bit in each pattern.

    Returns:
        list: A list of test patterns with bits flipped according to the given probability.
    """
    test_patterns = []
    for pattern in patterns:
        test_pattern = [int(not value) if random.random() < probability else value for value in pattern]
        test_patterns.append(test_pattern)
    return test_patterns

def hamming_distance(pattern1, pattern2):
    """
    Calculates the Hamming distance between two strings of equal length.

    Args:
        pattern1 (str): The first string.
        pattern2 (str): The second string.

    Returns:
        int: The number of positions at which the corresponding symbols are different.
    """
    return sum(p1 != p2 for p1, p2 in zip(pattern1, pattern2))

def run_simulation(hopfield_network, patterns, test_patterns, max_iterations=100):
    """
    Runs a simulation of a Hopfield network given a set of patterns and test patterns.

    Parameters:
    hopfield_network (HopfieldNetwork): The Hopfield network to simulate.
    patterns (list): A list of patterns to train the network on.
    test_patterns (list): A list of patterns to test the network on.
    max_iterations (int): The maximum number of iterations to run for each test pattern.

    Returns:
    list: A list of tuples containing the Hamming distance, number of iterations, and energy for each test pattern.
    """
    results = []

    for test_pattern, target_pattern in zip(test_patterns, patterns):
        pattern = test_pattern.copy()
        prev_pattern = None
        iterations = 0

        while prev_pattern != pattern and iterations < max_iterations:
            prev_pattern = pattern.copy()
            pattern = hopfield_network.update(pattern)
            iterations += 1

        hamming_dist = hamming_distance(target_pattern, pattern)
        energy = hopfield_network.energy(pattern)
        results.append((hamming_dist, iterations, energy))

    return results

# A list of six floating-point values representing probabilities.
probabilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Iterate over a list of probabilities and run a simulation for each probability.
# The probabilities are used to generate test patterns with varying levels of noise.
# The simulation is run 5 times for each probability.
for probability in probabilities:
    for _ in range(5):
        test_patterns = generate_test_patterns(patterns, probability)
        run_simulation(hopfield_network, patterns, test_patterns)

def generate_random_patterns(n_patterns, n_units):
    """
    Generates a list of random patterns with the specified number of patterns and units.

    Args:
        n_patterns (int): The number of patterns to generate.
        n_units (int): The number of units in each pattern.

    Returns:
        A list of n_patterns patterns, each containing n_units units. Each unit is randomly set to 0 or 1.
    """
    return [[1 if random.random() < 0.5 else 0 for _ in range(n_units)] for _ in range(n_patterns)]

# Generate 3 random binary patterns of length 16.
# Set the probability of each bit being flipped to 0.2.
# Generate 5 test patterns with noise using the random patterns and the given probability.
# Run a simulation with the Hopfield network using the random patterns and the test patterns.
random_patterns = generate_random_patterns(3, 16)

probability = 0.2

for _ in range(5):
    test_patterns = generate_test_patterns(random_patterns, probability)
    run_simulation(hopfield_network, random_patterns, test_patterns)

# Add two more random patterns and test
random_patterns += generate_random_patterns(2, 16)

for _ in range(5):
    test_patterns = generate_test_patterns(random_patterns, probability)
    run_simulation(hopfield_network, random_patterns, test_patterns)

base_pattern = [1] * 8 + [0] * 8

def generate_related_patterns(base_pattern, n_patterns, probability):
    """
    Generates a list of related patterns based on a given base pattern.

    Args:
        base_pattern (str): The base pattern to generate related patterns from.
        n_patterns (int): The number of related patterns to generate.
        probability (float): The probability of each character in the related patterns being different from the base pattern.

    Returns:
        list: A list of related patterns generated from the base pattern.
    """
    return [generate_test_patterns([base_pattern], probability)[0] for _ in range(n_patterns)]

related_patterns = generate_related_patterns(base_pattern, 6, 0.125)

probability = 0

for _ in range(5):
    test_patterns = generate_test_patterns(related_patterns, probability)
    run_simulation(hopfield_network, related_patterns, test_patterns)

patterns = random_patterns 

hopfield_network = HopfieldNetwork(16)
hopfield_network.train(patterns)

def run_simulation_async(hopfield_network, patterns, test_patterns, async_update=True, max_iterations=100):
    """
    Runs an asynchronous simulation on a Hopfield network for a given set of test patterns.

    Args:
        hopfield_network (HopfieldNetwork): The Hopfield network to simulate.
        patterns (List[np.ndarray]): The patterns that the network has been trained on.
        test_patterns (List[np.ndarray]): The patterns to test the network on.
        async_update (bool, optional): Whether to use asynchronous or synchronous updates. Defaults to True.
        max_iterations (int, optional): The maximum number of iterations to run for each test pattern. Defaults to 100.

    Returns:
        List[Tuple[int, int, float]]: A list of tuples containing the Hamming distance, number of iterations, and energy for each test pattern.
    """
    results = []

    for test_pattern, target_pattern in zip(test_patterns, patterns):
        pattern = test_pattern.copy()
        prev_pattern = None
        iterations = 0

        while prev_pattern != pattern and iterations < max_iterations:
            prev_pattern = pattern.copy()
            pattern = hopfield_network.update(pattern, async_update=async_update)
            iterations += 1

        hamming_dist = hamming_distance(target_pattern, pattern)
        energy = hopfield_network.energy(pattern)
        results.append((hamming_dist, iterations, energy))

    return results

probability = 0.2

for _ in range(5):
    test_patterns = generate_test_patterns(patterns, probability)
    run_simulation_async(hopfield_network, patterns, test_patterns, async_update=True)  # Asynchronous updating
    run_simulation_async(hopfield_network, patterns, test_patterns, async_update=False)  # Synchronous updating

def print_results(results):
    """
    Prints the results of a simulated annealing algorithm run.

    Args:
        results (list): A list of tuples containing the Hamming distance, number of iterations, and energy for each run.

    Returns:
        None
    """
    print("Hamming Distance | Iterations | Energy")
    for hamming_dist, iterations, energy in results:
        print(f"{hamming_dist:^15} | {iterations:^10} | {energy}")

# Part 1
print("Part 1 Results:")
for probability in probabilities:
    for _ in range(5):
        test_patterns = generate_test_patterns(patterns, probability)
        results = run_simulation(hopfield_network, patterns, test_patterns)
        print_results(results)

# Part 2
print("\nPart 2 Results:")
probability = 0.2
for _ in range(5):
    test_patterns = generate_test_patterns(random_patterns, probability)
    results = run_simulation(hopfield_network, random_patterns, test_patterns)
    print_results(results)

# Part 3
print("\nPart 3 Results:")
probability = 0
for _ in range(5):
    test_patterns = generate_test_patterns(related_patterns, probability)
    results = run_simulation(hopfield_network, related_patterns, test_patterns)
    print_results(results)

# Part 4
print("\nPart 4 Results:")
probability = 0.2
for _ in range(5):
    test_patterns = generate_test_patterns(patterns, probability)
    print("Asynchronous updating:")
    results = run_simulation_async(hopfield_network, patterns, test_patterns, async_update=True)
    print_results(results)
    print("Synchronous updating:")
    results = run_simulation_async(hopfield_network, patterns, test_patterns, async_update=False)
    print_results(results) 
