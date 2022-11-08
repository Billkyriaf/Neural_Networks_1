#include <iostream>

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
    return 0;
}
