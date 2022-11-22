#include <iostream>

#include "Network.h"
#include "mnist/MNIST_Import.h"

int main() {
    // Import the MNIST dataset
    MNIST_Import mnist(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte",
            "data/t10k-images.idx3-ubyte",
            "data/t10k-labels.idx1-ubyte"
    );

    //Start importing the images and labels
    std::cout << std::endl << "Importing the images and labels..." << std::endl << std::endl;

    mnist.readMetadata();  // Read the metadata from the file
    mnist.printMetadata();  // Print the metadata


    std::vector<MNIST_Image *> training_images;  // The vector to store the training images
    training_images.reserve(mnist.getTrDataCount());  // Reserve the memory for better performance

    mnist.readTrainingData(training_images);  // Read the training data


    std::vector<MNIST_Image *> test_images;  // The vector to store the test images
    test_images.reserve(mnist.getTsDataCount());  // Reserve the memory for better performance

    mnist.readTestData(test_images);  // Read the test data

    // Create the network
    std::vector<int> layers = {784, 16, 16, 10};
    Network network(int(layers.size()), 0.002, 50, layers, "ReLU", "Xavier", training_images, test_images);

//    network.printNetwork();

    network.trainNetwork();

    network.testNetwork();

    return 0;
}
