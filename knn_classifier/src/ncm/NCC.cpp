#include <algorithm>
#include "NCC.h"

#define MEAN_THREADS 16

NCC::NCC(const std::vector<MNIST_Image *> &training_images, const std::vector<MNIST_Image *> &test_images) {
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

    // initialize the class means
    for (int i = 0; i < 10; i++) {
        auto *image = new MNIST_Image(i);
        class_means[i] = image;
    }

    this->class_counts.fill(0);
}

NCC::NCC(const std::vector<MNIST_Image *>& training_images, const std::vector<MNIST_Image *>& test_images,
         const std::array<MNIST_Image *, 10>& means, std::array<int, 10> counts) {
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

    // Deep copy the means
    for (int i = 0; i < 10; i++) {
        auto *image = new MNIST_Image(*means[i]);

        class_means[i] = image;
    }

    this->class_counts = counts;
}

NCC::~NCC() {
    // Free the memory
    for (auto & training_image : training_images) {
        delete training_image;
    }

    for (auto & test_image : test_images) {
        delete test_image;
    }

    for (auto & mean : class_means) {
        delete mean;
    }
}

void NCC::incrementCorrect() {
    n_correct++;
    n_tests++;
}

void NCC::incrementIncorrect() {
    n_incorrect++;
    n_tests++;
}

typedef struct {
    int start;  // The index of the first image to process
    int end;  // The index of the last image to process

    std::vector<std::vector<int>> *means;   // The means
    std::vector<int> *counts;  // The counts

    NCC *ncm;  // The NCC object
} Thread_args;


void *calculateMeansThread(void *args) {
    auto *thread_args = (Thread_args *) args;

    for (int i = thread_args->start; i < thread_args->end; i++) {
        int label = thread_args->ncm->training_images[i]->getLabel();

        // update the count
        thread_args->counts->at(label)++;

        // update the mean
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            thread_args->means->at(label)[j] += thread_args->ncm->training_images[i]->getPixel(j);
        }
    }

    return nullptr;
}

void NCC::calculateMeans() {

    std::vector<std::vector<int>> thread_counts;
    thread_counts.reserve(MEAN_THREADS);

    for (int i = 0; i < MEAN_THREADS; i++) {
        std::vector<int> thread_count;
        thread_count.reserve(10);

        for (int j = 0; j < 10; j++) {
            thread_count.push_back(0);
        }

        thread_counts.push_back(thread_count);
    }

    std::vector<std::vector<std::vector<int>>> thread_means;
    thread_means.reserve(MEAN_THREADS);

    for (int i = 0; i < MEAN_THREADS; i++) {
        std::vector<std::vector<int>> thread_mean;
        thread_mean.reserve(10);

        for (int j = 0; j < 10; j++) {
            std::vector<int> mean;
            mean.reserve(MNIST_IMAGE_SIZE);

            for (int k = 0; k < MNIST_IMAGE_SIZE; k++) {
                mean.push_back(0);
            }

            thread_mean.push_back(mean);
        }

        thread_means.push_back(thread_mean);
    }

    typedef void * (*thread_function_ptr)(void *);

    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    std::array<Thread_args, MEAN_THREADS> thread_args{};
    std::array<pthread_t, MEAN_THREADS> threads{};

    int n_images = int(training_images.size());
    int n_images_per_thread = n_images / MEAN_THREADS;

    // Create the threads
    for (int i = 0; i < MEAN_THREADS; i++) {
        thread_args[i].start = i * n_images_per_thread;
        thread_args[i].end = (i + 1) * n_images_per_thread;
        thread_args[i].means = &thread_means[i];
        thread_args[i].counts = &thread_counts[i];
        thread_args[i].ncm = this;

        pthread_create(&threads[i], &pthread_custom_attr, (thread_function_ptr) calculateMeansThread, &thread_args[i]);
    }

    // Wait for the threads to finish
    for (int i = 0; i < MEAN_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    // Combine the results to the first thread
    for (int thread_id = 1; thread_id < MEAN_THREADS; thread_id++) {
        for (int label = 0; label < 10; label++) {
            thread_counts[0][label] += thread_counts[thread_id][label];

            for (int pixel = 0; pixel < MNIST_IMAGE_SIZE; pixel++) {
                thread_means[0][label][pixel] += thread_means[thread_id][label][pixel];
            }
        }
    }

    // Calculate the means
    for (int label = 0; label < 10; label++) {
        for (int pixel = 0; pixel < MNIST_IMAGE_SIZE; pixel++) {
            class_means[label]->setPixel(thread_means[0][label][pixel] / thread_counts[0][label], pixel);
        }
    }

    // Set the counts
    for (int i = 0; i < 10; i++) {
        class_counts[i] = thread_counts[0][i];
    }

    // free the memory
    pthread_attr_destroy(&pthread_custom_attr);
}

void NCC::classifyImage(int test_index, bool verbose) {
    std::array<double, 10> class_distances{};  // The distance from the test image to each class mean image

    // Calculate the distance from the test image to the class mean images
    for (int i = 0; i < 10; ++i) {
        class_distances[i] = class_means.at(i)->calculateDistance(*test_images.at(test_index));
    }

    int min_label;  // The label of the class mean image with the smallest distance
    double min = class_distances[0];  // The smallest distance

    // Find the class mean image with the smallest distance
    for (int i = 1; i < 10; ++i) {
        if (class_distances[i] < min) {
            min = class_distances[i];
            min_label = i;
        }
    }

    // Update the stats
    if (test_images.at(test_index)->isLabel(min_label)) {
        incrementCorrect();
    } else {
        incrementIncorrect();
    }

    // Print the results
    if (verbose) {
        std::cout << "Test image " << test_index << " is a " << int(test_images.at(test_index)->getLabel()) << std::endl;
        std::cout << "Test image " << test_index << " classified as " << min_label << std::endl;
        std::cout << "Votes: " << std::endl;

        for (int i = 0; i < 10; ++i) {
            std::cout << i << ": " << class_distances.at(i) << std::endl;
        }

    } else {
        if (test_images.at(test_index)->isLabel(min_label)) {
            std::cout << "Test image " << test_index << " classified correctly as " << min_label << std::endl;
        } else {
            std::cout << "Test image " << test_index << " classified incorrectly as " << min_label << ". It was "
            << int(test_images.at(test_index)->getLabel()) <<std::endl;
        }
    }
}

void NCC::printStats() {
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

void NCC::calculateAccuracy() {
    accuracy = (double(n_correct) / double(n_tests)) * 100;
}

std::array<MNIST_Image *, 10> NCC::getClassMeans() const {
    return class_means;
}


