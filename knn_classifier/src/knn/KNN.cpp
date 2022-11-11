#include <algorithm>
#include <pthread.h>

#include "KNN.h"

#define N_THREADS 16

/**
 * Class constructor
 *
 * @param k                 The number of nearest neighbors to use
 * @param training_images   The training images
 * @param test_images       The test images
 */
KNN::KNN(int k, const std::vector<MNIST_Image *>& training_images, const std::vector<MNIST_Image *>& test_images) {
    KNN::k = k;
    KNN::n_tests = 0;
    KNN::n_correct = 0;
    KNN::n_incorrect = 0;

    KNN::training_images.reserve(training_images.size());
    KNN::test_images.reserve(test_images.size());

    // Deep copy the training images
    for (auto & training_image : training_images) {
        auto *image = new MNIST_Image(*training_image);

        KNN::training_images.push_back(image);
    }

    // Deep copy the test images
    for (auto & test_image : test_images) {
        auto *image = new MNIST_Image(*test_image);

        KNN::test_images.push_back(image);
    }
}

/**
 * Class destructor
 */
KNN::~KNN() {
    // Free the memory
    for (auto & training_image : KNN::training_images) {
        delete training_image;
    }

    for (auto & test_image : KNN::test_images) {
        delete test_image;
    }
}

/**
 * Increment the number of correct classifications
 */
void KNN::incrementCorrect() {
    KNN::n_correct++;
    KNN::n_tests++;
}

/**
 * Increment the number of incorrect classifications
 */
void KNN::incrementIncorrect() {
    KNN::n_incorrect++;
    KNN::n_tests++;
}


/**
 * Struct to pass arguments to the thread
 */
typedef struct {
    int start;       // The index of the first image to process
    int end;         // The index of the last image to process

    int test_index;  // The index of the test image
    KNN *knn;        // The KNN object
} Thread_args;


/**
 * Thread function for each classifier. Classifies the images in the range [start, end)
 *
 * @param arg The thread arguments
 * @return nullptr
 */
void *calculateDistancesThread(void *args) {
    auto *thread_args = (Thread_args *) args;
    KNN *knn = thread_args->knn;
    int test_index = thread_args->test_index;

    for (int i = thread_args->start; i < thread_args->end; i++) {
        // Reset the distance. Useful if the same training images are used for multiple test images
        knn->training_images.at(i)->calculateDistance(*knn->test_images.at(test_index));
    }

    return nullptr;
}

/**
 * Classifies the test image at the given index
 *
 * @param test_index  The index of the test image
 * @param verbose     Whether to print the classification result
 * @return The predicted label
 */
int KNN::classifyImage(int test_index, bool verbose) {
    typedef void * (*thread_function_ptr)(void *);  // Pointer to a thread function

    pthread_attr_t pthread_custom_attr;       // Custom attributes for the threads
    pthread_attr_init(&pthread_custom_attr);  // Initialize the custom attributes

    std::array<Thread_args, N_THREADS> thread_args{};  // The arguments for each thread
    std::array<pthread_t, N_THREADS> threads{};        // The threads

    // Calculate the number of images to process per thread
    int n_images = int(training_images.size());
    int n_images_per_thread = n_images / N_THREADS;

    // Create the threads
    for (int i = 0; i < N_THREADS; ++i) {
        thread_args[i].start = i * n_images_per_thread;
        thread_args[i].end = (i + 1) * n_images_per_thread;
        thread_args[i].test_index = test_index;
        thread_args[i].knn = this;

        // Create the thread
        pthread_create(&threads[i], &pthread_custom_attr, (thread_function_ptr)calculateDistancesThread, &thread_args[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < N_THREADS; ++i) {
        pthread_join(threads[i], nullptr);
    }


    std::sort(training_images.begin(), training_images.end(), [](MNIST_Image *a, MNIST_Image *b) {
        return a->getDistance() < b->getDistance();
    });


    // Count the number of images with each label
    std::array<int, 10> label_count {};
    for (int i = 0; i < k; ++i) {
        label_count.at(training_images.at(i)->getLabel())++;
    }

    // Find the label with the most votes
    int max_label = 0;
    int max_count = 0;
    for (int i = 0; i < 10; ++i) {
        if (label_count.at(i) > max_count) {
            max_label = i;
            max_count = label_count.at(i);
        }
    }

    // Update the stats
    if (test_images.at(test_index)->isLabel(max_label)) {
        incrementCorrect();
    } else {
        incrementIncorrect();
    }

    // Print the results
    if (verbose) {
        std::cout << "Test image " << test_index << " is a " << int(test_images.at(test_index)->getLabel()) << std::endl;
        std::cout << "Test image " << test_index << " classified as " << max_label << std::endl;
        std::cout << "Votes: " << std::endl;

        for (int i = 0; i < 10; ++i) {
            std::cout << i << ": " << label_count.at(i) << std::endl;
        }

    } else {
//        if (test_images.at(test_index)->isLabel(max_label)) {
//            std::cout << "Test image " << test_index << " classified correctly as " << max_label << std::endl;
//        } else {
//            std::cout << "Test image " << test_index << " classified incorrectly as " << max_label << ". It was "
//            << int(test_images.at(test_index)->getLabel()) <<std::endl;
//        }
    }

    // Return the predicted label
    return max_label;
}

/**
 * Calculates the accuracy of the classifier
 */
void KNN::calculateAccuracy() {
    accuracy = (double(n_correct) / double(n_tests)) * 100;
}

/**
 * Prints the accuracy of the classifier
 */
void KNN::printStats(){
    calculateAccuracy();  // Calculate the accuracy

    // Print the results
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Number of tests: " << n_tests << std::endl;
    std::cout << "Number of correct classifications: " << n_correct << std::endl;
    std::cout << "Number of incorrect classifications: " << n_incorrect << std::endl;
    std::cout.precision(3);
    std::cout << "Accuracy: " << std::fixed << accuracy << "%" << std::endl;

}

/**
 * Accumulates the stats from another KNN object
 *
 * @param knn_classifiers The KNN vector to accumulate stats from
 */
void KNN::accumulateStats(const std::vector<KNN *>& knn_classifiers) {
    // Accumulate the stats
    for (auto & knn_classifier : knn_classifiers) {
        if (knn_classifier != this) {
            n_correct += knn_classifier->n_correct;
            n_incorrect += knn_classifier->n_incorrect;
            n_tests += knn_classifier->n_tests;
        }
    }
}
