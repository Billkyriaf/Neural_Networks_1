#ifndef KNN_CLASSIFIER_NCC_H
#define KNN_CLASSIFIER_NCC_H


#include <array>
#include <vector>

#include "../mnist/MNIST_Image.h"

class NCC {
public:
    // Constructors
    NCC() = delete;
    NCC(const std::vector<MNIST_Image *>& training_images, const std::vector<MNIST_Image *>& test_images);
    NCC(const std::vector<MNIST_Image *>& training_images, const std::vector<MNIST_Image *>& test_images,
        const std::array<MNIST_Image *, 10>& means, std::array<int, 10> counts);

    // Destructor
    ~NCC();

    // Getters
    std::array<MNIST_Image *, 10> getClassMeans() const;

    // Setters
    void incrementCorrect();
    void incrementIncorrect();

    // Functions
    void calculateMeans();
    int classifyImage(int test_index, bool verbose = false);
    void printStats();

    // Friend functions
    friend void * calculateMeansThread(void *arg);

private:
    // Variables
    std::array<MNIST_Image *, 10> class_means{};  /// The mean vector of each class
    std::array<int, 10> class_counts {};          /// The number of images in each class

    std::vector<MNIST_Image *> cluster_means{};          /// The mean vector of each cluster
    std::vector<std::vector<MNIST_Image *>> clusters{};  /// The clusters of images

    std::vector<MNIST_Image *> training_images;   /// The training images
    std::vector<MNIST_Image *> test_images;       /// The training images

    int n_clusters {0};     /// The number of clusters
    int n_tests {0};        /// The number of tests performed
    int n_correct {0};      /// The number of correct classifications
    int n_incorrect {0};    /// The number of incorrect classifications
    double accuracy {0};    /// The accuracy of the classifier

    // Functions
    void calculateAccuracy();
};


#endif
