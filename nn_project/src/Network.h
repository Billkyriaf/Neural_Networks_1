#ifndef NN_PROJECT_NETWORK_H
#define NN_PROJECT_NETWORK_H

#include <vector>

#include "mnist/MNIST_Image.h"
#include "perceptrons/Perceptron.h"

// TODO : Add performance metrics

class Network {
public:
    // Constructors
    Network()= delete;
    Network (int n_layers, double l_rate, int epochs, std::vector<int>& n_perceptrons,
             const std::string& activation_function, const std::string& initialization_function,
             std::vector<MNIST_Image *>& training_set, std::vector<MNIST_Image *>& test_set);

    // Destructor
    ~Network();

    // Getters

    // Setters

    // Functions
    void trainNetwork();
    void testNetwork();
    void printNetwork() const;

private:
    std::vector<MNIST_Image *> training_images {};   // Training images
    std::vector<MNIST_Image *> test_images {};       // Test images

    int n_layers {0};                              // Number of layers
    std::vector<int> layers_sizes {0};             // Number of neurons in each layer

    std::vector<std::vector<Perceptron *>> network;     // Layers of the network

    std::vector<Perceptron *> *output_layer {nullptr};  // Output layer of the network
    std::vector<Perceptron *> *input_layer {nullptr};   // Input layer of the network

    std::vector<double *> outputs {};                   // Outputs of the network

    std::string activation_function {};                 // Activation function of the network
    std::string initialization_function {};             // Initialization function of the network

    double learning_rate {0};                      // Learning rate of the network
    int n_epochs {0};                              // Number of epochs

    std::array<double, 10> expected_output {};     // Expected output of the network for a given image

    // Functions
    void initializeNetwork(std::vector<int>& n_perceptrons);
    void inputImage(MNIST_Image *image);
    void backPropagate();
};


#endif
