import numpy as np
from sklearn.datasets import load_iris
from minisom import MiniSom
from sklearn.metrics import pairwise_distances
from itertools import combinations
import time

start_time = time.time()
# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# Perform SOM clustering
som_rows = 10
som_cols = 10
som = MiniSom(som_rows, som_cols, data.shape[1], sigma=0.3, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 1000)

# Get the predicted labels
predicted_labels = np.zeros(data.shape[0])
for i, sample in enumerate(data):
    winner = som.winner(sample)
    predicted_labels[i] = winner[0] * som_cols + winner[1]

# Calculate the RAND index
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
rand_index = calculate_rand_index(target, predicted_labels)
end_time = time.time()
execution_time = end_time - start_time

print("RAND Index:", rand_index)
print(f"Execution time : {execution_time} seconds")