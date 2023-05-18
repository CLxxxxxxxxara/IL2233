#include <cmath>
#include <cstdlib>
#include <limits>
#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>
void readDataFromFile(const std::string &filename, double *data, double *true_assignment, int n_samples, int m_features)
{
    std::ifstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < n_samples; i++)
        {
            file >> true_assignment[i];
            for (int j = 0; j < m_features; j++)
            {
                file >> data[i * m_features + j];
            }
        }
        file.close();
    }
    else
    {
        std::cout << "Failed to open file: " << filename << std::endl;
    }
}

struct t_pos
{
    int x;
    int y;
};

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

// Function to find the best-matching unit (BMU) for a given data point
t_pos findBMU(const std::vector<std::vector<double>> &grid, const double *sample, int m_features, int height, int width)
{
    t_pos bmu;
    double min_distance = std::numeric_limits<double>::max();

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            const std::vector<double> &weight = grid[i * width + j];
            // 计算weight这个vector和当前sample指向的指针之间的distance
            double distance = euclideanDistance(weight.data(), sample, m_features);

            if (distance < min_distance)
            {
                min_distance = distance;
                bmu.x = i;
                bmu.y = j;
            }
        }
    }

    return bmu;
}

// Function to update the weights of the SOM grid
void updateWeights(std::vector<std::vector<double>> &grid, const double *sample, int m_features, int height, int width, const t_pos &bmu, float lr, float sigma)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::vector<double> &weight = grid[i * width + j];
            double distance = std::sqrt((i - bmu.x) * (i - bmu.x) + (j - bmu.y) * (j - bmu.y));
            double influence = std::exp(-distance / (2 * sigma * sigma));

            for (int k = 0; k < m_features; ++k)
            {
                weight[k] += lr * influence * (sample[k] - weight[k]);
            }
        }
    }
}

// Function to perform SOM clustering
void SOM(t_pos *assignment, double *data, int n_samples, int m_features, int height, int width, int max_iter, float lr, float sigma)
{
    std::vector<std::vector<double>> grid(height * width, std::vector<double>(m_features));

    // Initialize the weights of the SOM grid randomly
    std::srand(std::time(0));
    for (int i = 0; i < height * width; ++i)
    {
        for (int j = 0; j < m_features; ++j)
        {
            grid[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    // Perform SOM updates
    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Update the learning rate and sigma
        float current_lr = lr * (1 - iter / max_iter);
        float current_sigma = sigma * (1 - iter / max_iter);

        // Iterate over the samples
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx)
        {
            // point to the memory address of the current
            const double *sample = &data[sample_idx * m_features];
            t_pos bmu = findBMU(grid, sample, m_features, height, width);
            assignment[sample_idx] = bmu; // Assign the sample to the BMU position
            updateWeights(grid, sample, m_features, height, width, bmu, current_lr, current_sigma);
        }
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
//     // Test dataset
//     const int n_samples = 10;
//     const int m_features = 2;
//     double data[n_samples * m_features] = {
//         1.0, 1.0,
//         1.5, 2.0,
//         3.0, 4.0,
//         5.0, 7.0,
//         3.5, 5.0,
//         4.5, 5.0,
//         3.5, 4.5,
//         2.5, 1.5,
//         3.0, 2.5,
//         2.0, 7.0};

//     // SOM parameters
//     const int height = 3;
//     const int width = 3;
//     const int max_iter = 100;
//     const float lr = 0.1;
//     const float sigma = 1.0;

//     // SOM clustering
//     t_pos assignment[n_samples];
//     SOM(assignment, data, n_samples, m_features, height, width, max_iter, lr, sigma);

//     // Print the clustering results
//     std::cout << "Sample\tAssigned Position (Row, Col)\n";
//     for (int i = 0; i < n_samples; ++i)
//     {
//         std::cout << i << "\t(" << assignment[i].x << ", " << assignment[i].y << ")\n";
//     }

//     return 0;
// }

int main()
{
    // Define the dataset properties
    int n_samples = 180;  // Number of samples
    int m_features = 128; // Number of features

    // Read the dataset from a file
    const std::string filename = "BME_TEST.txt";
    double *data = new double[n_samples * m_features];
    double *true_assignment = new double[n_samples];
    readDataFromFile(filename, data, true_assignment, n_samples, m_features);

    // Define the SOM parameters
    int height = 1;     // SOM grid height
    int width = 3;      // SOM grid width
    int max_iter = 100; // Maximum number of iterations
    float lr = 0.1;     // Learning rate
    float sigma = 1.0;  // Sigma value

    // Perform SOM clustering
    t_pos *assignment = new t_pos[n_samples];
    SOM(assignment, data, n_samples, m_features, height, width, max_iter, lr, sigma);
    int *som_cluster_assignment = new int[n_samples];
    int int_true[n_samples];
    for (int i = 0; i < n_samples; i++)
    {
        int_true[i]=static_cast<int>(true_assignment[i]);
        som_cluster_assignment[i] = assignment[i].y; 
    }
    // Print the clustering results
    for (int i = 0; i < n_samples; i++)
    {
        std::cout << "Sample " << i << " assigned to position (" << assignment[i].x << ", " << assignment[i].y << ")" << std::endl;
    }
    double randIndex = calculateRandIndex(som_cluster_assignment,int_true,n_samples);
    std::cout << "RAND Index: " << randIndex << std::endl;
    // Clean up
    delete[] assignment;
    delete[] true_assignment;
    delete[] data;

    return 0;
}