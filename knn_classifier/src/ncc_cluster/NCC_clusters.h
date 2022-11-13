#ifndef KNN_CLASSIFIER_NCC_CLUSTERS_H
#define KNN_CLASSIFIER_NCC_CLUSTERS_H


#include <vector>
#include <random>
#include "../mnist/MNIST_Image.h"

class NCC_clusters {
public:
    // Constructors
    NCC_clusters(int n_clusters, const std::vector<MNIST_Image *>& training_images,
                 const std::vector<MNIST_Image *>& test_images);

    NCC_clusters(const std::string& cluster_dir, const std::vector<MNIST_Image *>& training_images,
                 const std::vector<MNIST_Image *>& test_images);

    NCC_clusters() = delete;

    // Destructor
    ~NCC_clusters();

    // Getters

    // Setters
    void incrementCorrect();
    void incrementIncorrect();

    // Functions
    int classifyImage(int test_index, bool verbose = false);
    void printStats();
    void printClusterCounts(int cluster_index);
    void saveMeanClusters();

    // Friend functions
    friend void *centroidThread(void *arg);

private:
    // Variables
    std::vector<MNIST_Image *> cluster_means{};           /// The mean vector of each cluster
    std::vector<std::vector<MNIST_Image *>> clusters{};   /// The clusters of images

    std::vector<MNIST_Image *> training_images;   /// The training images
    std::vector<MNIST_Image *> test_images;       /// The training images

    std::default_random_engine *generator;        /// The random number generator

    int n_clusters {0};     /// The number of clusters
    int n_tests {0};        /// The number of tests performed
    int n_correct {0};      /// The number of correct classifications
    int n_incorrect {0};    /// The number of incorrect classifications
    double accuracy {0};    /// The accuracy of the classifier
    bool from_file {false}; /// Whether the clusters were loaded from a file

    // Functions
    void initializeCentroids(int dataset_fraction = 30);
    void fitClusters(bool is_final = false, int dataset_fraction = 60);
    void updateClusterMean(int cluster_index, MNIST_Image *image, int n_images);
//    void detectConvergence(int cluster_index);
    void determineClusterLabel();
    void calculateAccuracy();
};

#endif
