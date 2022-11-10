#ifndef KNN_CLASSIFIER_KNN_H
#define KNN_CLASSIFIER_KNN_H


#include <cstdint>
#include <vector>
#include "../mnist/MNIST_Image.h"


class KNN {
public:
    // Constructors
    KNN() = default;

    KNN(int k, const std::vector<MNIST_Image *>& training_images, const std::vector<MNIST_Image *>& test_images);

    // Destructor
    ~KNN();

    // Getters

    // Setters
    void incrementCorrect();
    void incrementIncorrect();

    // Functions
    bool classifyImage(int test_index, bool verbose = false);
    void printStats();
    void accumulateStats(const std::vector<KNN *>& knn_classifiers);
    friend void * calculateDistancesThread(void *arg);


private:
    // Variables
    uint32_t k {1};  /// The number of nearest neighbors to consider
    std::vector<MNIST_Image *> training_images;   /// The training images
    std::vector<MNIST_Image *> test_images;       /// The training images

    int n_tests {0};        /// The number of tests performed
    int n_correct {0};      /// The number of correct classifications
    int n_incorrect {0};    /// The number of incorrect classifications
    double accuracy {0};    /// The accuracy of the classifier

    // Functions
    void calculateAccuracy();
};


#endif
