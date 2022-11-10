#include <iostream>
#include <cmath>

#include "MNIST_Import.h"
#include "KNN.h"

#define CLASS_THREADS 16

typedef struct {
    int thread_id;   // The thread ID
    int start;       // The index of the first image to process
    int end;         // The index of the last image to process

    KNN *knn;        // The KNN object
} thread_data;

void *classify(void *arg) {
    auto *data = (thread_data *) arg;

    for (int i = data->start; i < data->end; i++) {
        data->knn->classifyImage(i, false);

        if (i % 100 == 0 && data->thread_id == 0 && i != 0) {
            std::cout << "Every thread classified approximately " << i << " images" << std::endl;
        }
    }

    return nullptr;
}

int main() {
    MNIST_Import mnist(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte",
            "data/t10k-images.idx3-ubyte",
            "data/t10k-labels.idx1-ubyte"
    );

    mnist.readMetadata();  // Read the metadata from the file
    mnist.printMetadata();

    // The vector to store the training images
    std::vector<MNIST_Image *> training_images;
    training_images.reserve(mnist.getTrDataCount());

    mnist.readTrainingData(training_images);  // Read the training data

    // The vector to store the test images
    std::vector<MNIST_Image *> test_images;
    test_images.reserve(mnist.getTsDataCount());

    mnist.readTestData(test_images);  // Read the test data

    // Save the first image from the training set and the first image from the test set as pgm files to confirm that
    // the data was read correctly
    std::cout << "Saving training image as 'train_0.pgm'... Label: " << (int)training_images.at(0)->getLabel() << std::endl;
    save_image("images/train_0", training_images.at(0)->getPixels());

    std::cout << "Saving test image as 'test_1.pgm'... Label: " << (int)test_images.at(0)->getLabel() << std::endl;
    save_image("images/test_0", test_images.at(0)->getPixels());

    // Create a KNN classifiers
    std::vector<KNN *> classifiers;
    classifiers.reserve(CLASS_THREADS);

    for (int i = 0; i < CLASS_THREADS; i++) {
        classifiers.push_back(new KNN(3, training_images, test_images));
    }

    // Create the threads
    pthread_t threads[CLASS_THREADS];
    thread_data data[CLASS_THREADS];

    for (int i = 0; i < CLASS_THREADS; i++) {
        data[i].thread_id = i;
        data[i].start = i * int(test_images.size() / CLASS_THREADS);
        data[i].end = (i + 1) * int(test_images.size() / CLASS_THREADS);
        data[i].knn = classifiers.at(i);

        pthread_create(&threads[i], nullptr, classify, &data[i]);
    }

    // Wait for the threads to finish
    for (unsigned long thread : threads) {
        pthread_join(thread, nullptr);
    }


    classifiers.at(0)->accumulateStats(classifiers);
    classifiers.at(0)->printStats();

    // Free the memory
    for (int i = 0; i < mnist.getTrDataCount(); ++i) {
        delete training_images.at(i);
    }

    for (int i = 0; i < mnist.getTsDataCount(); ++i) {
        delete test_images.at(i);
    }

    return 0;
}
