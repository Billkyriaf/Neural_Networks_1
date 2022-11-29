#include "Network.h"

#include <utility>
#include <random>

#include "../include/progressbar.h"
#include "network_functions/activation_functions.h"
#include "network_functions/initialization_functions.h"

#define BATCH_SIZE 600

// TODO : Detect convergence
// TODO : Test the network after each epoch with a small subset of the test data
// TODO : Display stats after each epoch

/**
 * Constructor of the Network class. The constructor initializes the network with the given parameters. The activation
 * function of the output layer is always softmax
 *
 * @param n_layers                 Number of layers of the network
 * @param l_rate                   Learning rate of the network
 * @param epochs                   Number of epochs of the network
 * @param n_perceptrons            Number of perceptrons in each layer
 * @param activation_function      Activation algorithm of the network
 * @param initialization_function  Initialization algorithm of the network
 * @param training_set             Training dataset of the network
 * @param test_set                 Test dataset of the network
 */
Network::Network(int n_layers, double l_rate, int epochs, std::vector<int>& n_perceptrons,
                 const std::string& activation_function, const std::string& initialization_function,
                 std::vector<MNIST_Image *> &training_set, std::vector<MNIST_Image *> &test_set) : n_layers(n_layers),
                 layers_sizes(n_perceptrons), learning_rate(l_rate), n_epochs(epochs){

    // Deep copy of the training and test sets  // TODO optimize this not to deep copy
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

/**
 * Destructor of the Network class. The destructor deletes the allocated perceptrons.
 */
Network::~Network() {
    // Delete the perceptrons
    for (int i = 0; i < Network::n_layers; i++) {
        for (int j = 0; j < Network::layers_sizes.at(i); j++) {
            delete Network::network[i][j];
        }
    }
}

// ====================== Functions ======================

/**
 * Function to train the network. The function partitions the training set into mini-batches and trains the network.
 * The training is done for the number of epochs specified in the constructor. For each epoch the training set batch is
 * created with a new random order using a uniform distribution.
 */
void Network::trainNetwork() {
    // Create a random generator
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()

    // Create a uniform distribution
    std::uniform_int_distribution<> dis(0, int(Network::training_images.size()) - 1);

    std::vector<int> image_indexes;  // Vector to store the indexes of the images in the batch

    std::cout << "Training the network..." << std::endl;

    // for each batch
    for (int epoch = 0; epoch < Network::n_epochs; epoch++) {

        // Create the batch
        for (int i = 0; i < BATCH_SIZE; i++) {
            image_indexes.push_back(dis(gen));
        }

        // Save the first image of the batch
//        Network::training_images[image_indexes[0]]->saveImage("batch_" + std::to_string(epoch), 255);

        std::cout << "    Epoch: " << epoch << "  ";
        progressbar bar(BATCH_SIZE);  // Progress bar to display the progress of the training

        // for each image in the batch
        for (auto &image_index : image_indexes) {
            bar.update();  // Update the progress bar

            // 1. input the image to the network...
            Network::inputImage(Network::training_images[image_index]);

            // ... and get the label
            int label = Network::training_images[image_index]->getLabel();

            // 2. Call the activation function of every layer starting from the input layer
            for (int layer_idx = 0; layer_idx < Network::n_layers; layer_idx++) {

                // call the activation function of every perceptron in the layer
                for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[layer_idx]; perceptron_idx++) {
                    Network::network[layer_idx][perceptron_idx]->activate();
                }
            }

            // 3. Use softmax to get the output layer values ...
            std::vector<double> softmax_outputs = Softmax(Network::outputs);

            // and set the output layer values
            for (int i = 0; i < Network::layers_sizes[Network::n_layers - 1]; i++) {
                Network::network[Network::n_layers - 1][i]->setOutput(softmax_outputs[i]);
            }

            // 4. Update the error of the output layer
            for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[Network::n_layers - 1]; perceptron_idx++) {

                // The output layer has 10 perceptrons, one for each digit
                if (perceptron_idx == label) {
                    //If the index of the perceptron is equal to the label of the image then the target value is 1 ...
                    (*Network::output_layer)[perceptron_idx]->updateError(1.0);

                } else {
                    // ...otherwise it is 0
                    (*Network::output_layer)[perceptron_idx]->updateError(0.0);
                }

                (*Network::output_layer)[perceptron_idx]->updateGradient();
            }

            // 5. Update the errors and gradients of all the layers using the error of the output layer

            // For each layer starting from the last hidden layer ....
            for (int layer_idx = Network::n_layers - 2; layer_idx > 0; --layer_idx) {

                // ... and for each perceptron in each layer ...
                for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[layer_idx]; perceptron_idx++) {

                    auto & perceptron = Network::network[layer_idx][perceptron_idx];  // reference to the perceptron

                    std::vector<double> next_layer_error; // vector of the errors of the next layer

                    // 5.1 get the errors of the next layer
                    for (int next_idx = 0; next_idx < Network::layers_sizes[layer_idx + 1]; next_idx++) {
                        auto & next_layer_perceptron = Network::network[layer_idx + 1][next_idx];
                        next_layer_error.push_back(next_layer_perceptron->getError());
                    }


                    std::vector<double> next_layer_weights;  // vector of the weights of the next layer

                    // 5.2 get the weights of the next layer that connect to the current perceptron
                    for (int next_idx = 0; next_idx < Network::layers_sizes[layer_idx + 1]; next_idx++) {
                        auto & next_layer_perceptron = Network::network[layer_idx + 1][next_idx];
                        next_layer_weights.push_back(next_layer_perceptron->getWeight(perceptron_idx));
                    }

                    // 5.3 update the error of the perceptron
                    perceptron->updateError(0.0, next_layer_error, next_layer_weights);

                    // 5.4 update the gradient of the perceptron
                    perceptron->updateGradient();
                }
            }
        }

        std::cout << std :: endl;

        // 6. Once the batch is finished, update the weights and biases of the network
        Network::backPropagate();

        // 7. Clear the image indexes and repeat
        image_indexes.clear();

        // 8. Print the accuracy of the network every 10 epochs
        if ((epoch + 1) % 10 == 0 && epoch != 0 && epoch != Network::n_epochs - 1) {
            Network::testNetwork();
        }
    }

    std::cout << std::endl << std::endl << "Resulted network: " << std::endl;
    Network::testNetwork();
}

/**
 * Function to test the network. The function passes the hole training set to the network.
 */
void Network::testNetwork() {
    int n_correct = 0;
    int n_wrong = 0;

    std::cout << std::endl <<  "        Testing the network:  ";

    progressbar bar(int(Network::test_images.size()));  // Progress bar to display the progress of the training

    // for each image in the training set
    for (auto & test_image : Network::test_images) {
        bar.update();  // Update the progress bar

        // 1. input the image to the network ...
        Network::inputImage(test_image);

        // ... and get the label
        int label = test_image->getLabel();

        // 2. Call the activation function of every layer starting from the input layer
        for (int layer_idx = 0; layer_idx < Network::n_layers; layer_idx++) {

            // call the activation function of every perceptron in the layer
            for (int perceptron_idx = 0; perceptron_idx < Network::layers_sizes[layer_idx]; perceptron_idx++) {
                Network::network[layer_idx][perceptron_idx]->activate();
            }
        }

        // 3. Use softmax to get the output layer values ...
        std::vector<double> softmax_outputs = Softmax(Network::outputs);

        // 3. Get the index of the maximum output
        int max_idx = 0;
        for (int i = 1; i < softmax_outputs.size(); i++) {
            if (softmax_outputs[i] > softmax_outputs[max_idx]) {
                max_idx = i;
            }
        }

        // 4. check if the output is correct
        if (max_idx == label) {
            n_correct++;
        } else {
            n_wrong++;
        }
    }
    std::cout << std::endl;

    std::cout << "        Correct: " << n_correct << std::endl;
    std::cout << "        Wrong: " << n_wrong << std::endl;
    std::cout << "        Accuracy: " << double(n_correct) / (n_correct + n_wrong) << std::endl << std::endl;
}

/**
 * Print information about the network and the perceptrons
 */
void Network::printNetwork() const {
    std::cout << std::endl << "Network: " << std::endl;
    std::cout << "    Layers (" <<  Network::n_layers << "):  Input layer " << Network::layers_sizes[0] << " -> ";
    for (int i = 1; i < Network::n_layers - 1; i++) {
        std::cout << Network::layers_sizes[i] << " -> ";
    }
    std::cout << "Output layer " << Network::layers_sizes[Network::n_layers - 1] << std::endl << std::endl;


    std::cout << "    Activation function:     " << Network::activation_function << std::endl;
    std::cout << "    Initialization function: " << Network::initialization_function << std::endl << std::endl;
    std::cout << "    Learning rate:       " << Network::learning_rate << std::endl;
    std::cout << "    Training Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "    Number of epochs:    " << Network::n_epochs << std::endl << std::endl << std::endl;


//    std::cout << std::endl << "Network:" << std::endl;
//    for (int i = 0; i < Network::n_layers; i++) {
//        std::cout << "    Layer " << i << std::endl;
//        for (int j = 0; j < Network::layers_sizes[i]; j++) {
//            std::cout << "        Perceptron " << j << std::endl;
//            Network::network[i][j]->printPerceptron();
//        }
//    }
}

/**
 * Initialize the network with the given parameters
 *
 * @param n_perceptrons Vector of the number of perceptrons in each layer
 */
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

    // Set the activation function of each perceptron except the input and the output layer
    for (int i = 1; i < n_layers - 1; i++) {
        for (int j = 0; j < n_perceptrons[i]; j++) {
            Network::network[i][j]->setActivationFunction(activation_function_ptr);
            Network::network[i][j]->setActivationFunctionDerivative(derivative_function_ptr);
        }
    }

    // set the activation function derivative of the output layer
    for (int i = 0; i < n_perceptrons[n_layers - 1]; i++) {
        Network::network[n_layers - 1][i]->setActivationFunctionDerivative(SoftmaxDerivative);
    }

    // Initialize the weights and biases of each perceptron.
    if (Network::initialization_function == "Xavier"){
        XavierInitialization(Network::network);

    } else if (Network::initialization_function == "Kaiming"){
        KaimingInitialization(Network::network);

    } else if (Network::initialization_function == "Zero"){
        ZeroInitialization(Network::network);
    }
}

/**
 * Pass an image to the input layer. The image pixes are already normalized.
 *
 * @param image  The image to pass to the input layer
 */
void Network::inputImage(MNIST_Image *image) {

    // Set the input of the first layer to the image pixels
    for (int i = 0; i < Network::input_layer->size(); i++) {
        (*Network::input_layer)[i]->setInput(image->getPixelPtr(i), 0);
    }
}

/**
 * Update the weights and biases of the network using the backpropagation algorithm. Every perceptron has already
 * calculated the sum of the gradients of the weights and biases so we must just update the weights and biases
 * based on the calculated gradients.
 */
void Network::backPropagate() {
    // For each layer...
    for (int layer_idx = 1; layer_idx < Network::n_layers; layer_idx++) {

        // ... and each perceptron in the layer...
        for (int perc_idx = 0; perc_idx < Network::layers_sizes[layer_idx]; perc_idx++) {

            // ... update the weights and biases
            Network::network[layer_idx][perc_idx]->changeWeightsAndBias(Network::learning_rate);
        }
    }
}





