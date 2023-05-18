import numpy as np
from minisom import MiniSom
from itertools import combinations
import time
start_time = time.time()
# Load dataset
data = np.loadtxt('BME_TEST.txt')

# Extract true labels from dataset
true_labels = data[:, 0]

# Extract features from dataset
features = data[:, 1:]



# Set SOM grid size
grid_rows = 10
grid_cols = 10

# Initialize SOM
som = MiniSom(grid_rows, grid_cols, features.shape[1], sigma=0.3, learning_rate=0.5)

# Train SOM
som.train(features, 100)

# Get predicted labels
predicted_labels = np.zeros(features.shape[0])
for i, sample in enumerate(features):
    winner = som.winner(sample)
    predicted_labels[i] = winner[0] * grid_cols + winner[1]

def calculate_rand_index(true_labels, predicted_labels):
    n_samples = len(true_labels)
    tp_plus_tn = 0
    n = n_samples * (n_samples - 1) / 2

    # Calculate the number of true positives (TP) and true negatives (TN)
    for i, j in combinations(range(n_samples), 2):
        if (true_labels[i] == true_labels[j]) and (predicted_labels[i] == predicted_labels[j]):
            tp_plus_tn += 1
        if (true_labels[i] != true_labels[j]) and (predicted_labels[i] != predicted_labels[j]):
            tp_plus_tn += 1


    # Calculate the RAND index
    rand_index = tp_plus_tn / n
    return rand_index


# Compute the RAND index
rand_index = calculate_rand_index(true_labels, predicted_labels)

end_time = time.time()
execution_time = end_time - start_time

print("RAND Index: {:.4f}".format(rand_index))
print(f"Execution time : {execution_time} seconds")