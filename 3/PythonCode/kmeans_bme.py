import os
import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
import time
start_time = time.time()
# Set OMP_NUM_THREADS environment variable to suppress the warning
os.environ['OMP_NUM_THREADS'] = '1'

# Load dataset
data = np.loadtxt('BME_TEST.txt')

# Extract true labels from dataset
true_labels = (data[:, 0]-1)

# Extract features from dataset
features = data[:, 1:]



# Set number of clusters for K-means
n_clusters = 3
    #= len(np.unique(true_labels))

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters)
predicted_labels = kmeans.fit_predict(features)

# Calculate RAND Index to evaluate accuracy
#rand_index = adjusted_rand_score(true_labels, predicted_labels)
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
