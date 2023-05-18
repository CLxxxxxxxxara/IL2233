from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from itertools import combinations
import numpy as np
import time
start_time = time.time()
# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target


# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
predicted_labels = kmeans.fit_predict(data)

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