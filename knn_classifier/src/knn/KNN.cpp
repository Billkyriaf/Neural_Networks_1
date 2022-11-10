#include <algorithm>
#include <pthread.h>

#include "KNN.h"

#define N_THREADS 16


KNN::KNN(int k, const std::vector<MNIST_Image *>& training_images, const std::vector<MNIST_Image *>& test_images) {
    this->k = k;
    this->n_tests = 0;
    this->n_correct = 0;
    this->n_incorrect = 0;

    this->training_images.reserve(training_images.size());
    this->test_images.reserve(test_images.size());

    // Deep copy the training images
    for (auto & training_image : training_images) {
        auto *image = new MNIST_Image(*training_image);

        this->training_images.push_back(image);
    }

    // Deep copy the test images
    for (auto & test_image : test_images) {
        auto *image = new MNIST_Image(*test_image);

        this->test_images.push_back(image);
    }
}

KNN::~KNN() {
    // Free the memory
    for (auto & training_image : training_images) {
        delete training_image;
    }

    for (auto & test_image : test_images) {
        delete test_image;
    }
}

void KNN::incrementCorrect() {
    n_correct++;
    n_tests++;
}


void KNN::incrementIncorrect() {
    n_incorrect++;
    n_tests++;
}

typedef struct {
    int start;       // The index of the first image to process
    int end;         // The index of the last image to process

    int test_index;  // The index of the test image
    KNN *knn;        // The KNN object
} Thread_args;

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


bool KNN::classifyImage(int test_index, const bool verbose) {
    typedef void * (*thread_function_ptr)(void *);

    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    std::array<Thread_args, N_THREADS> thread_args{};
    std::array<pthread_t, N_THREADS> threads{};

    int n_images = int(training_images.size());
    int n_images_per_thread = n_images / N_THREADS;

    for (int i = 0; i < N_THREADS; ++i) {

        thread_args[i].start = i * n_images_per_thread;
        thread_args[i].end = (i + 1) * n_images_per_thread;
        thread_args[i].test_index = test_index;
        thread_args[i].knn = this;

        pthread_create(&threads[i], &pthread_custom_attr,
                       reinterpret_cast<void *(*)(void *)>(calculateDistancesThread), &thread_args[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < N_THREADS; ++i) {
        pthread_join(threads[i], nullptr);
    }


    // Sort the training images by distance
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

//    if (test_index % 100 == 0) {
//        std::cout << "Tested " << test_index << " images" << std::endl;
//    }

    // Return true if the label is correct
    return test_images.at(test_index)->isLabel(max_label);
}


void KNN::calculateAccuracy() {
    accuracy = (double(n_correct) / double(n_tests)) * 100;
}


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
