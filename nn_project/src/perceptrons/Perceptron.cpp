#include <iostream>
#include "Perceptron.h"


Perceptron::Perceptron(int n_inputs,  bool input_layer,  bool output_layer) : is_input(input_layer), is_output(output_layer) {
    // Initialize weights and inputs
    Perceptron::weights.reserve(n_inputs);
    Perceptron::inputs.reserve(n_inputs);

    for (int i = 0; i < n_inputs; ++i) {
        Perceptron::weights.push_back(0);
        Perceptron::inputs.push_back(nullptr);
        Perceptron::weight_grad_sum.push_back(0);
    }
}


// Getters
double Perceptron::getOutput() const {
    return Perceptron::output;
}

double *Perceptron::getOutputPtr() {
    return &(Perceptron::output);
}

int Perceptron::getNInputs() const {
    return int(Perceptron::weights.size());
}

// Setters
void Perceptron::setWeights(std::vector<double>& set_weights) {
    Perceptron::weights = set_weights;
}

void Perceptron::setWeight(double weight, int index) {
    Perceptron::weights[index] = weight;
}

void Perceptron::setInputs(std::vector<double *> &set_inputs) {
    Perceptron::inputs = set_inputs;
}

void Perceptron::setInput(double *input, int index) {
    Perceptron::inputs[index] = input;
}

void Perceptron::setBias(double set_bias) {
    Perceptron::bias = set_bias;
}

void Perceptron::setActivationFunction(
        void (*activation_function)(double &, std::vector<double *> &, std::vector<double> &, double)) {
    Perceptron::activationFunction = activation_function;
}

void Perceptron::setActivationFunctionDerivative(double (*activation_function_derivative)(double)) {
    Perceptron::activationFunctionDerivative = activation_function_derivative;
}

/**
 * Print information about the perceptron
 */
void Perceptron::printPerceptron() const {
    std::cout << "            Weights: ";
    for (double weight : Perceptron::weights) {
        std::cout << weight << " ";
    }

    std::cout << std::endl;
    std::cout << "            Bias: " << Perceptron::bias << std::endl;
}

/**
 * The forward propagation function of the perceptron. If the perceptron is an input perceptron, the output is simply the
 * input. Otherwise, the activation function is called.
 *
 */
void Perceptron::activate() {
    // If the perceptron is an input perceptron, the output is simply the input
    if (Perceptron::is_input){
        Perceptron::output = *(Perceptron::inputs[0]);

    } else if (Perceptron::activationFunction != nullptr) {  // else call the activation function
        Perceptron::activationFunction(Perceptron::output, Perceptron::inputs, Perceptron::weights, Perceptron::bias);

    } else {
        std::cerr << "Error: activation function not set" << std::endl;
    }

    // update the number of samples
    Perceptron::n_samples++;
}

void Perceptron::testPerceptron() {
    // pass the input to the output
    Perceptron::output = *(Perceptron::inputs[0]);
}

double Perceptron::getBias() const {
    return Perceptron::bias;
}

double Perceptron::getWeight(int index) const {
    return Perceptron::weights[index];
}

void Perceptron::updateWeight(double delta, int index) {
    Perceptron::weights[index] -= delta;
}

void Perceptron::updateBias(double delta) {
    Perceptron::bias -= delta;
}

void Perceptron::updateError(double target, const std::vector<double>& next_layer_errors,
                             const std::vector<double>& next_layer_weights) {

    if (Perceptron::is_output) {
        /*
         * Update the average error sum  δ = (y_hat - y) * f'(z) where y_hat is the output of the perceptron, y is the
         * expected output and f'(z) is the derivative of the activation function and z is the weighted sum of the
         * inputs
         */

        Perceptron::error = (target - Perceptron::output) * Perceptron::activationFunctionDerivative(Perceptron::output);

    } else {
        /*
         * back propagate the error. The formula is δ = Σ(w_i * δ_i) * f' where w_i is the weight of the i-th input,
         * δ_i is the error of the i-th input OF THE NEXT LAYER, f' is the derivative of the activation function.
         */

        double error_sum = 0;
        for (int i = 0; i < next_layer_errors.size(); ++i) {
            error_sum += next_layer_weights[i] * next_layer_errors[i];
        }

        Perceptron::error = error_sum * Perceptron::activationFunctionDerivative(Perceptron::output);
    }
}

const std::vector<double> &Perceptron::getWeights() const {
    return Perceptron::weights;
}

double Perceptron::getError() const {
    return Perceptron::error;
}

void Perceptron::updateGradient() {
    // update the gradient sum for the weights
    for (int i = 0; i < Perceptron::weights.size(); ++i) {
        Perceptron::weight_grad_sum[i] += Perceptron::error * *(Perceptron::inputs[i]);
    }

    // update the gradient sum for the bias
    Perceptron::bias_grad_sum += Perceptron::error;
}

void Perceptron::changeWeightsAndBias(double learning_rate) {
    // update the weights
    for (int i = 0; i < Perceptron::weights.size(); ++i) {
        Perceptron::weights[i] -= learning_rate * (Perceptron::weight_grad_sum[i] / Perceptron::n_samples);
    }

    // update the bias
    Perceptron::bias -= learning_rate * (Perceptron::bias_grad_sum / Perceptron::n_samples);

    // reset the gradient sums
    Perceptron::weight_grad_sum = std::vector<double>(Perceptron::weights.size(), 0);
    Perceptron::bias_grad_sum = 0;
    Perceptron::n_samples = 0;
}


