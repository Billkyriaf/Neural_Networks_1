#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <random>

#include "initialization_functions.h"


/**
 * Xavier initialization of the weights. It is a good initialization for the sigmoid activation functions.
 *
 * @param network   Network to initialize
 */
void XavierInitialization(std::vector<std::vector<Perceptron *>> &network){
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()

    for (auto & layer : network) {
        for (auto & perceptron : layer) {

            // Normal distribution
            int fan_in = perceptron->getNInputs();
            int fan_out = int(layer.size());

            double fan_avg = (fan_in + fan_out) / 2.0;

            std::normal_distribution<> dis(0, sqrt(1.0 / fan_avg));

            // Set the weights to the generated values and the bias to 0
            for (int k = 0; k < perceptron->getNInputs(); ++k) {
                perceptron->setWeight(dis(gen), k);
                perceptron->setBias(0);
            }
        }
    }
}


/**
 * Initialize the weights of the network to all zeros
 *
 * @param network   Network to initialize
 */
void ZeroInitialization(std::vector<std::vector<Perceptron *>> &network){
    for (auto & layer : network) {
        for (auto & perceptron : layer) {

            // Set all weights and biases to zero
            for (int k = 0; k < perceptron->getNInputs(); ++k) {
                perceptron->setWeight(0, k);
                perceptron->setBias(0);
            }
        }
    }
}


/**
 * Kaiming initialization of the weights. The kaiming initialization initializes the weights with a normal distribution
 * with mean 0 and standard deviation sqrt(2 / number of inputs). Best used with ReLU activation function.
 *
 * @param network   Network to initialize
 */
void KaimingInitialization(std::vector<std::vector<Perceptron *>> &network) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()

    for (auto &layer: network) {
        for (auto & perceptron: layer) {
            // Generate a normal distribution with mean 0 and standard deviation sqrt(2 / input_size)
            std::normal_distribution<> dist(0.0, sqrt(2.0 / perceptron->getNInputs()));

            // Set the weights to the generated values and the bias to 0  TODO: Should the bias be initialized to 0?
            for (int k = 0; k < perceptron->getNInputs(); ++k) {
                perceptron->setWeight(dist(gen), k);
                perceptron->setBias(0);
            }
        }
    }
}