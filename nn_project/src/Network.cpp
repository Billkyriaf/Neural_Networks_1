#include "Network.h"

#include <utility>
#include "network_functions/activation_functions.h"
#include "network_functions/initialization_functions.h"

Network::Network(int n_layers, double l_rate, int epochs, std::vector<int>& n_perceptrons,
                 const std::string& activation_function, const std::string& initialization_function,
                 std::vector<MNIST_Image *> &training_set, std::vector<MNIST_Image *> &test_set) : n_layers(n_layers),
                 layers_sizes(n_perceptrons), learning_rate(l_rate), n_epochs(epochs){

    // Deep copy of the training and test sets
    Network::training_images.reserve(training_set.size());
    Network::test_images.reserve(test_set.size());

    // Deep copy the training images
    for (auto &training_image : training_set) {
        auto *image = new MNIST_Image(*training_image);

        Network::training_images.push_back(image);
    }

    // Deep copy the test images
    for (auto & test_image : test_set) {
        auto *image = new MNIST_Image(*test_image);

        Network::test_images.push_back(image);
    }

    // Set the activation function
    if (activation_function == "ReLU" || activation_function == "Sigmoid") {
        Network::activation_function = activation_function;

    } else {
        std::cout << "Activation function not recognized defaulting to ReLU" << std::endl;
        Network::activation_function = "ReLU";
    }

    // Set the initialization function
    if (initialization_function == "Xavier" || initialization_function == "Normalized" ||
    initialization_function == "Kaiming" || initialization_function == "Zeros"){

        Network::initialization_function = initialization_function;

    } else {

        // Default initialization depends on the activation function
        if (activation_function == "Sigmoid") {
            std::cout << "Initialization function not recognized defaulting to Xavier" << std::endl;
            Network::initialization_function = "Sigmoid";
        }
        else {
            std::cout << "Initialization function not recognized defaulting to Kaiming" << std::endl;
            Network::initialization_function = "Kaiming";
        }
    }

    // Initialize the network
    initializeNetwork(n_perceptrons);
}

Network::~Network() {
    // Delete the perceptrons
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < layers_sizes[i]; j++) {
            delete Network::network[i][j];
        }
    }
}

void Network::initializeNetwork(std::vector<int>& n_perceptrons) {
    // Reserve space for the layers
    Network::network.reserve(Network::n_layers);

    // create the layers
    for (int i = 0; i < Network::n_layers; i++) {
        std::vector<Perceptron *> layer;
        for (int j = 0; j < n_perceptrons[i]; j++) {
            if (i == 0) {
                layer.push_back(new Perceptron(1, true));  // this is the input layer
            } else if (i != Network::n_layers - 1) {
                layer.push_back(new Perceptron(n_perceptrons[i - 1]));
            } else {
                layer.push_back(new Perceptron(n_perceptrons[i - 1], false, true));  // this is the output layer
            }
        }
        Network::network.push_back(layer);
    }

    // Set the input and output layers for easy access
    Network::input_layer = &Network::network[0];
    Network::output_layer = &Network::network[Network::n_layers - 1];

    /*
     * Connect the input of each perceptron to the output of the previous layer. The only unconnected perceptrons are
     * the input of the first layer and the output of the last layer. The input of the first layer is set to the image
     * vector and the output of the last layer is the prediction.
     */
    for (int layer_idx = 1; layer_idx < Network::n_layers; layer_idx++) {
        for (int perc_idx = 0; perc_idx < n_perceptrons[layer_idx]; perc_idx++) {
            for (int k = 0; k < n_perceptrons[layer_idx - 1]; k++) {
                Network::network[layer_idx][perc_idx]->setInput(Network::network[layer_idx - 1][k]->getOutputPtr(), k);
            }
        }
    }

    // Set the outputs
    Network::outputs.reserve(n_perceptrons[Network::n_layers - 1]);

    for (int i = 0; i < n_perceptrons[Network::n_layers - 1]; i++) {
        Network::outputs.push_back(Network::network[Network::n_layers - 1][i]->getOutputPtr());
    }

    // Set the activation function and the derivative for every perceptron
    void (*activation_function_ptr)(double &, std::vector<double *> &, std::vector<double> &, double);
    double (*derivative_function_ptr)(double);

    if (activation_function == "ReLU"){
        activation_function_ptr = ReLU;
        derivative_function_ptr = ReLUDerivative;

    } else if (activation_function == "Sigmoid"){
        activation_function_ptr = Sigmoid;
        derivative_function_ptr = SigmoidDerivative;
    }

    // Set the activation function of each perceptron
    for (int i = 1; i < n_layers; i++) {
        for (int j = 0; j < n_perceptrons[i]; j++) {
            Network::network[i][j]->setActivationFunction(activation_function_ptr);
            Network::network[i][j]->setActivationFunctionDerivative(derivative_function_ptr);
        }
    }

    // Initialize the weights and biases of each perceptron.
    if (Network::initialization_function == "Xavier"){
        XavierInitialization(Network::network);

    } else if (Network::initialization_function == "Normalized Xavier"){
        NormalizedXavierInitialization(Network::network);

    } else if (Network::initialization_function == "Kaiming"){
        KaimingInitialization(Network::network);

    } else if (Network::initialization_function == "Zero"){
        ZeroInitialization(Network::network);
    }
}

/**
 * Print information about the network and the perceptrons
 */
void Network::printNetwork() const {
    std::cout << std::endl << "Number of layers: " << Network::n_layers << std::endl;
    std::cout << "    Layers sizes: " << "Input layer " << Network::layers_sizes[0] << " -> ";
    for (int i = 1; i < Network::n_layers - 1; i++) {
        std::cout << Network::layers_sizes[i] << " -> ";
    }
    std::cout << "Output layer " << Network::layers_sizes[Network::n_layers - 1] << std::endl << std::endl;


    std::cout << "Activation function: " << Network::activation_function << std::endl;
    std::cout << "Initialization function: " << Network::initialization_function << std::endl;

    std::cout << std::endl << "Network:" << std::endl;
    for (int i = 0; i < Network::n_layers; i++) {
        std::cout << "    Layer " << i << std::endl;
        for (int j = 0; j < Network::layers_sizes[i]; j++) {
            std::cout << "        Perceptron " << j << std::endl;
            Network::network[i][j]->printPerceptron();
        }
    }
}

int Network::getNLayers() const {
    return Network::n_layers;
}

std::vector<int> Network::getLayersSizes() const {
    return Network::layers_sizes;
}

void Network::setInputs(MNIST_Image *image) {
    // Set the input of the first layer to the image vector
    for (int i = 0; i < Network::input_layer->size(); i++) {
        (*Network::input_layer).at(i)->setInput(image->getPixelPtr(i), 0);
    }
}

std::vector<double *> Network::getOutputs() const {
    return Network::outputs;
}

void Network::backpropagation() {
    // Every perceptron has already calculated the sum of the gradients of the weights and biases so we must just update
    // the weights and biases of each perceptron

    for (int layer_idx = 1; layer_idx < Network::n_layers; layer_idx++) {
        for (int perc_idx = 0; perc_idx < Network::layers_sizes[layer_idx]; perc_idx++) {

            Network::network[layer_idx][perc_idx]->changeWeightsAndBias(Network::learning_rate);
        }
    }
}

void Network::trainNetwork() {
    // break the training in batches
    int n_batches = Network::n_epochs;
    int batch_size = int(Network::training_images.size()) / n_batches;

    // for each batch
    for (int batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        // for each image in the batch
        for (int image_idx = batch_idx * batch_size; image_idx < (batch_idx + 1) * batch_size; image_idx++) {
            // set the inputs
            Network::setInputs(Network::training_images[image_idx]);
            int label = Network::training_images[image_idx]->getLabel();

            // call the activation function of every layer starting from the input layer
            for (int layer_idx = 0; layer_idx < Network::n_layers; layer_idx++) {

                // call the activation function of every perceptron in the layer
                for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[layer_idx]; perceptron_idx++) {
                    Network::network[layer_idx][perceptron_idx]->activate();
                }
            }

            // update the error of the output layer
            for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[Network::n_layers - 1]; perceptron_idx++) {
                if (perceptron_idx == label) {
                    Network::output_layer->at(perceptron_idx)->updateError(1.0);
                } else {
                    Network::output_layer->at(perceptron_idx)->updateError(0.0);
                }
            }

            // update the errors and gradients of all the layers from the output layer to the input layer
            for (int layer_idx = Network::n_layers - 2; layer_idx > 0; --layer_idx) {
                // for each perceptron in the layer
                for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[layer_idx]; perceptron_idx++) {

                    auto & perceptron = Network::network[layer_idx][perceptron_idx];  // reference to the perceptron

                    // vector of the errors of the next layer
                    std::vector<double> next_layer_error;
                    for (int next_idx = 0; next_idx < Network::layers_sizes[layer_idx + 1]; next_idx++) {
                        auto & next_layer_perceptron = Network::network[layer_idx + 1][next_idx];
                        next_layer_error.push_back(next_layer_perceptron->getError());
                    }

                    // vector of the weights of the next layer
                    std::vector<double> next_layer_weights;
                    for (int next_idx = 0; next_idx < Network::layers_sizes[layer_idx + 1]; next_idx++) {
                        auto & next_layer_perceptron = Network::network[layer_idx + 1][next_idx];
                        next_layer_weights.push_back(next_layer_perceptron->getWeight(perceptron_idx));
                    }

                    // update the error of the perceptron
                    perceptron->updateError(0.0, next_layer_error, next_layer_weights);

                    // update the gradient of the perceptron
                    perceptron->updateGradient();
                }
            }
        }

        // once the batch is finished, update the weights and biases of the network
        Network::backpropagation();
    }
}

void Network::testNetwork() {
    int n_correct = 0;
    int n_wrong = 0;

    for (auto & test_image : Network::test_images) {
        // set the inputs
        Network::setInputs(test_image);
        int label = test_image->getLabel();

        // call the activation function of every layer starting from the input layer
        for (int layer_idx = 0; layer_idx < Network::n_layers; layer_idx++) {

            // call the activation function of every perceptron in the layer
            for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[layer_idx]; perceptron_idx++) {
                Network::network[layer_idx][perceptron_idx]->activate();
            }
        }

        // get the output of the network
        Network::outputs = Network::getOutputs();

        // get the index of the maximum output
        int max_idx = 0;
        for (int i = 1; i < Network::outputs.size(); i++) {
            if (*Network::outputs[i] > *Network::outputs[max_idx]) {
                max_idx = i;
            }
        }

        // check if the output is correct
        if (max_idx == label) {
            n_correct++;
        } else {
            n_wrong++;
        }
    }

    std::cout << "Correct: " << n_correct << std::endl;
    std::cout << "Wrong: " << n_wrong << std::endl;
    std::cout << "Accuracy: " << double(n_correct) / (n_correct + n_wrong) << std::endl;
}
