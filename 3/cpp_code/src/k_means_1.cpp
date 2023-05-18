#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

void readDataFromFile(const std::string& filename, double* data, double* true_assignment, int n_samples, int m_features) {
    std::ifstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < n_samples; i++) {
            file >>true_assignment[i];
            for (int j = 0; j < m_features; j++) {
                file >> data[i * m_features + j];
            }
        }
        file.close();
    } else {
        std::cout << "Failed to open file: " << filename << std::endl;
    }
}

// Function to calculate the Euclidean distance between two data points
double euclideanDistance(const double *point1, const double *point2, int m_features)
{
    double distance = 0.0;
    for (int i = 0; i < m_features; ++i)
    {
        double diff = point1[i] - point2[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

// Function to initialize the centroids randomly
void initializeCentroids(std::vector<int> &assignment, int n_samples, int K)
{
    for (int i = 0; i < n_samples; ++i)
    {
        assignment[i] = std::rand() % K;
    }
}

// Function to update the centroids based on the assigned samples
void updateCentroids(const std::vector<std::vector<double>> &samples, const std::vector<int> &assignment,
                     int n_samples, int m_features, int K, std::vector<std::vector<double>> &centroids)
{
    std::vector<int> counts(K, 0);
    for (int i = 0; i < n_samples; ++i)
    {
        int cluster = assignment[i];
        ++counts[cluster]; // 每个cluster样本的数量
        for (int j = 0; j < m_features; ++j)
        {
            centroids[cluster][j] += samples[i][j]; // 每个cluster所有样本的和
        }
    }
    for (int i = 0; i < K; ++i)
    {
        if (counts[i] > 0)
        {
            for (int j = 0; j < m_features; ++j)
            {
                centroids[i][j] /= counts[i]; // 每个cluster的每一个样本特征的均值成为新的中心值
            }
        }
    }
}

// Function to perform K-means clustering
void kmeans(int *assignment, int K, int max_iter, int n_samples, int m_features, double *data)
{
    std::vector<std::vector<double>> samples(n_samples, std::vector<double>(m_features));
    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 0; j < m_features; ++j)
        {
            samples[i][j] = data[i * m_features + j];
        }
    }

    std::vector<std::vector<double>> centroids(K, std::vector<double>(m_features));
    std::vector<int> previous_assignment(n_samples);

    // Randomly initialize the centroids
    initializeCentroids(previous_assignment, n_samples, K);

    int iter = 0;
    while (iter < max_iter)
    {
        // Update the centroids based on the current assignment
        updateCentroids(samples, previous_assignment, n_samples, m_features, K, centroids);

        // Assign each sample to the nearest centroid
        for (int i = 0; i < n_samples; ++i)
        {
            double min_distance = 1000000.0;
            int best_cluster = -1;
            for (int j = 0; j < K; ++j)
            {
                double distance = euclideanDistance(samples[i].data(), centroids[j].data(), m_features);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    best_cluster = j;
                }
            }
            assignment[i] = best_cluster; // 把第i个sample分到最近的那个cluster中
            //std::cout << "sample " << i << " belongs to cluster " << best_cluster << std::endl;
        }

        // Check if the assignment has converged
        bool converged = true;
        for (int i = 0; i < n_samples; ++i)
        {
            if (assignment[i] != previous_assignment[i])
            {
                converged = false;
                break;
            }
        }

        // If the assignment has converged, exit the loop
        if (converged)
        {
            for (int i = 0; i < n_samples; ++i)
                //std::cout << assignment[i] << std::endl;
            break;
        }

        // Update the previous assignment for the next iteration
        for (int i = 0; i < n_samples; ++i)
        {
            previous_assignment[i] = assignment[i];
        }

        ++iter;
    }
}

double calculateRandIndex(const int* assignment1, const int* assignment2, int n_samples) {
    int TP = 0; // True positives
    int TN = 0; // True negatives
    int FP = 0; // False positives
    int FN = 0; // False negatives

    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            if (assignment1[i] == assignment1[j] && assignment2[i] == assignment2[j]) {
                ++TP;
            } else if (assignment1[i] != assignment1[j] && assignment2[i] != assignment2[j]) {
                ++TN;
            } else if (assignment1[i] == assignment1[j] && assignment2[i] != assignment2[j]) {
                ++FP;
            } else if (assignment1[i] != assignment1[j] && assignment2[i] == assignment2[j]) {
                ++FN;
            }
        }
    }

    double RI = static_cast<double>(TP + TN) / (TP + FP + TN + FN);
    return RI;
}
// int main()
// {
//     // Define the test dataset
//     const int n_samples = 10; // Number of samples
//     const int m_features = 2; // Number of features

//     double data[20] = {
//         2.0, 1.0,
//         2.5, 2.0,
//         1.5, 1.5,
//         3.5, 4.0,
//         4.0, 4.5,
//         1000.7, 30.3,
//         9.0, 10.0,
//         9.5, 9.5,
//         8.5, 9.0,
//         10.0, 8.5

//     };
//     int assignment[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

//     // Call the K-means function and validate the results
//     kmeans(assignment, 3, 100, n_samples, m_features, data);

//     return 0;
// }


int main() {
    // Define the dataset properties
    int n_samples = 180;  // Number of samples
    int m_features = 128;   // Number of features

    // Read the dataset from a file
    const std::string filename = "BME_TEST.txt";
    double* data = new double[n_samples * m_features];
    double* true_assignment = new double[n_samples];
    readDataFromFile(filename, data,true_assignment, n_samples, m_features);

    // Define the K-means parameters
    int K = 3;          // Number of clusters
    int max_iter = 100; // Maximum number of iterations

    // Perform K-means clustering
    int* assignment = new int[n_samples];
    kmeans(assignment, K, max_iter, n_samples, m_features, data);

    // Print the clustering results
    for (int i = 0; i < n_samples; i++) {
        std::cout << "Sample " << i << " assigned to cluster " << assignment[i] << std::endl;
    }

    
    int int_true[n_samples];
    for (int i = 0; i < n_samples; i++) {
        int_true[i] = static_cast<int>(true_assignment[i]);
    }

    double randIndex = calculateRandIndex(assignment, int_true, n_samples);
    std::cout << "RAND Index: " << randIndex << std::endl;

    // Clean up
    delete[] assignment;
    delete[] true_assignment;
    delete[] data;

    return 0;
}