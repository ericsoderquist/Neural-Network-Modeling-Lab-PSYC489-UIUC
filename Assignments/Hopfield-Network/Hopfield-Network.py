"""
Implementation of a Hopfield Network.
Author: Eric Soderquist
Class: PSYC 489: Neural Network Modeling Lab
"""
import numpy as np

class HopfieldNetwork:
    """
    A class representing a Hopfield Network.
    """
    def __init__(self, num_units: int):
        """
        Initialize the Hopfield Network.
        
        Args:
            num_units (int): The number of units in the network.
        """
        self.num_units = num_units
        self.weights = np.zeros((num_units, num_units))
                    new_state[i] = state[i]

            if np.array_equal(state, new_state):
                break

            state = new_state
            num_iterations += 1

        return state.tolist(), num_iterations, energy_list

def compute_hamming(self, pattern1, pattern2):
    return np.count_nonzero(np.array(pattern1) != np.array(pattern2))

def add_noise(pattern, p):
    """Add random noise to a pattern."""
    noise = np.random.choice([-1, 1], size=pattern.shape, p=[p, 1-p])
    return np.multiply(pattern, noise)

# Define patterns
patterns = [
    [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
    [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
    [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
]
patterns = np.array(patterns)
# Create Hopfield network and train it on the patterns
hopfield = HopfieldNetwork(16)
hopfield.train(patterns)

# Test the network
hamming_dists = []
num_iters_list = []
energy_list = []
max_iters = 100
num_fails = 0
for p in patterns:
    for p_change in [0, .1, .2, .3, .4, .5]:
        for run in range(5):
            input_pattern = add_noise(p, p_change)
            output_pattern, num_iterations, energy = hopfield.run(input_pattern, max_iters)
            num_iters_list.append(num_iterations)
            energy_list.append(energy)
            if not np.array_equal(p, output_pattern):
                num_fails += 1
            hamming_dist = hopfield.compute_hamming(p, output_pattern)
            hamming_dists.append(hamming_dist)

# Print summary statistics
print("Hamming distances:", hamming_dists)
print("Mean hamming distance:", np.mean(hamming_dists))
print("Standard deviation of hamming distances:", np.std(hamming_dists))
print("Number of times network failed to settle:", num_fails)
print("Number of iterations list:", num_iters_list)
print("Mean number of iterations:", np.mean(num_iters_list))
print("Standard deviation of iterations:", np.std(num_iters_list))
print("Energy list:", energy_list)

#2A)

# Define the size of the patterns
n_units = 16

# Initialize the training patterns as empty arrays
pattern1 = np.zeros(n_units)
pattern2 = np.zeros(n_units)
pattern3 = np.zeros(n_units)

# Activate each unit with probability 0.5
for i in range(n_units):
    if np.random.rand() > 0.5:
        pattern1[i] = 1
    if np.random.rand() > 0.5:
        pattern2[i] = 1
    if np.random.rand() > 0.5:
        pattern3[i] = 1

# Print the generated training patterns
print("Training pattern 1: ", pattern1)
print("Training pattern 2: ", pattern2)
print("Training pattern 3: ", pattern3)

#2B)

class HopfieldNet:
    def __init__(self, num_units):
        self.num_units = num_units
        self.weights = np.zeros((num_units, num_units))
        
    def train(self, patterns):
        # Validate input patterns
        if not isinstance(patterns, np.ndarray):
            raise TypeError('Patterns must be a NumPy array.')
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        
    def recall(self, test_patterns, max_iters=100, threshold=0):
        hamming_dists = []
        num_iters_list = []
        num_fails = 0
        energy_list = []
        
        for test_pattern in test_patterns:
            state = np.copy(test_pattern)
            for i in range(max_iters):
                energy = -0.5 * np.dot(np.dot(state, self.weights), state)
                energy_list.append(energy)
                new_state = np.zeros(self.num_units)
                for j in range(self.num_units):
                    activation = np.dot(self.weights[j], state)
                    if activation > threshold:
                        new_state[j] = 1
                    elif activation < -threshold:
                        new_state[j] = -1
                    else:
                        new_state[j] = state[j]
                if np.array_equal(state, new_state):
                    break
                state = new_state
            else:
                num_fails += 1
                
            hamming_dists.append(self.compute_hamming(new_state, test_pattern))
            num_iters_list.append(i+1)
        
        return hamming_dists, num_iters_list, energy_list, num_fails
    
    def compute_hamming(self, state1, state2):
        return np.sum(state1 != state2)

# Set the number of units in the patterns
num_units = 20

# Create three random training patterns
patterns = np.random.choice([-1, 1], size=(3, num_units), p=[0.5, 0.5])

# Create a Hopfield network with the given number of units
hopfield = HopfieldNet(num_units)

# Train the network on the training patterns
hopfield.train(patterns)

# Define the probabilities of changing each unit from 1 to -1 or vice versa
probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Set the maximum number of iterations for each test
max_iters = 100

# Set the number of times to run each test for each probability
num_runs = 5

# Initialize lists to store the results
hamming_dists = []
num_iters_list = []
num_fails = []

# Loop over each pattern in the training set
for p, pattern in enumerate(patterns):

    # Loop over each probability of changing each unit
    for q in probs:

        # Initialize a list to store the Hamming distances and number of iterations for each run
        hamming_dists_pq = []
        num_iters_pq = []

        # Run the test multiple times with random noise added each time
        for r in range(num_runs):

            # Add random noise to the pattern
            noisy_pattern = add_noise(pattern, q)

            # Run the network with the noisy pattern and record the number of iterations
            output_pattern, num_iters, energy_list = hopfield.run(noisy_pattern, max_iters=max_iters)

            # Compute the Hamming distance between the output pattern and the original pattern
            hamming_dist = compute_hamming(output_pattern, pattern)

            # Append the Hamming distance and number of iterations to the lists
            hamming_dists_pq.append(hamming_dist)
            num_iters_pq.append(num_iters)

        # Compute the mean Hamming distance and number of iterations for this pattern and probability
        mean_hamming_dist_pq = np.mean(hamming_dists_pq)
        mean_num_iters_pq = np.mean(num_iters_pq)

        # Append the mean Hamming distance and number of iterations to the overall lists
        hamming_dists.append(mean_hamming_dist_pq)
        num_iters_list.append(mean_num_iters_pq)

        # Count the number of times the network failed to settle
        num_fails_pq = hamming_dists_pq.count(num_units)  # If Hamming distance is equal to the number of units, network failed
        num_fails.append(num_fails_pq)

        # Print the results for this pattern and probability
        print(f"Pattern {p+1}, probability {q}:")

patterns = np.random.randint(2, size=(3, 100))
patterns[patterns == 0] = -1


q_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
num_runs = 5
max_iters = 1000

results = np.zeros((len(q_list), num_runs, 2))

for i, q in enumerate(q_list):
    for j in range(num_runs):
        noisy_patterns = add_noise(patterns, q)
        hopfield = HopfieldNetwork()
        hopfield.train(patterns)
        num_iters_list, num_iterations, hamming_dists, energy_list, num_fails = hopfield.run_patterns(
            noisy_patterns, max_iters)
        results[i, j, 0] = np.mean(hamming_dists)
        results[i, j, 1] = np.mean(num_iters_list)

        print(f"Q: {q}, run: {j}, Hamming Distance: {results[i, j, 0]}, Iterations: {results[i, j, 1]}, Failed: {num_fails}")


patterns2 = np.random.randint(2, size=(2, 100))
patterns2[patterns2 == 0] = -1

q = 0.2
num_runs = 5

for i in range(3, 5):
    patterns2 = np.concatenate((patterns2, np.random.randint(2, size=(1, 100))), axis=0)
    patterns2[patterns2 == 0] = -1
    noisy_patterns = add_noise(patterns2, q)
    hopfield = HopfieldNetwork()
    hopfield.train(patterns2)
    num_iters_list, num_iterations, hamming_dists, energy_list, num_fails = hopfield.run_patterns(noisy_patterns, max_iters)
    mean_hamming_dist = np.mean(hamming_dists)
    mean_num_iters = np.mean(num_iters_list)
    print(f"Patterns: {i+1}, Hamming Distance: {mean_hamming_dist}, Iterations: {mean_num_iters}, Failed: {num_fails}")

#3A

# create the base pattern
base_pattern = np.concatenate((np.ones((1,8)), np.zeros((1,8))), axis=1)

# create the training patterns
training_patterns = np.zeros((6,16))
for i in range(6):
    flip_indices = np.random.choice(16, size=2, replace=False, p=[0.125]*2)
    training_pattern = np.copy(base_pattern)
    training_pattern[:,flip_indices] = 1 - training_pattern[:,flip_indices]
    training_patterns[i,:] = training_pattern

#3B
# Define the base pattern
base_pattern = np.concatenate((np.ones(8), np.zeros(8)))

# Define the training patterns
training_patterns = []
for i in range(6):
    tp = np.copy(base_pattern)
    flip_indices = np.random.choice(16, 2, replace=False) # flip 2 indices with probability 0.125
    tp[flip_indices] = 1 - tp[flip_indices]
    training_patterns.append(tp)

# Create the network
network = HopfieldNet(16)

# Train the network on the training patterns
network.train(training_patterns)

# Test the network on the original training patterns
for tp in training_patterns:
    noisy_tp = add_noise(tp, 0)
    result, energy, num_iters = network.run_with_energy(noisy_tp, max_iters=100, threshold=1e-10)
    print("Original Pattern:\n", tp.reshape(4, 4))
    print("Noisy Pattern:\n", noisy_tp.reshape(4, 4))
    print("Result Pattern:\n", result.reshape(4, 4))
    print("Energy:", energy)
    print("Number of iterations:", num_iters)
    print("Hamming distance:", hamming_dist(result, tp))
    print("---------------------------")

"""When testing the network with the original Training patterns, it usually settles into one of the six stored patterns, 
even when there is noise in the input. This is because the Hopfield network is designed to converge to stored patterns that are closest 
to the input pattern in terms of the Hamming distance.
This phenomenon of the network settling into a stored pattern even when there is noise in the input can correspond to a form of pattern 
completion or recognition in a psychological sense. This is similar to how the human brain can recognize partially obscured or noisy 
stimuli and still identify them as familiar objects or patterns."""

#4A

# Define the training patterns
p1 = np.array([1, 1, -1, -1])
p2 = np.array([1, -1, 1, -1])
p3 = np.array([-1, -1, 1, 1])

# Create an instance of the Hopfield class and train the network
hopfield = HopfieldNet()
hopfield.train([p1, p2, p3])

#4B
#syncronous update
class HopfieldNetwork:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.weights = np.zeros((num_nodes, num_nodes))
        
    def train(self, patterns):
        # Validate input patterns
        if not isinstance(patterns, np.ndarray):
            raise TypeError('Patterns must be a NumPy array.')
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= len(patterns)
        np.fill_diagonal(self.weights, 0)
        
    def update(self, state):
        activations = np.dot(self.weights, state)
        new_state = np.where(activations > 0, 1, -1)
        return new_state

#asynchronous update
class HopfieldNetwork:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.weights = np.zeros((num_nodes, num_nodes))
        
    def train(self, patterns):
        # Validate input patterns
        if not isinstance(patterns, np.ndarray):
            raise TypeError('Patterns must be a NumPy array.')
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= len(patterns)
        np.fill_diagonal(self.weights, 0)
        
    def update(self, state, num_updates):
        indices = np.random.choice(self.num_nodes, num_updates, replace=False)
        activations = np.dot(self.weights[indices, :], state)
        new_state = np.copy(state)
        new_state[indices] = np.where(activations > 0, 1, -1)
        return new_state
