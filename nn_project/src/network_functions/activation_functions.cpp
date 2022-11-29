#include "activation_functions.h"

#include <cmath>

/**
 * Sigmoid activation function
 *
 * @param output   Output of the perceptron
 * @param inputs   Inputs of the perceptron
 * @param weights  Weights of the perceptron
 * @param bias     Bias of the perceptron
 */
void Sigmoid(double &output, std::vector<double *> &inputs, std::vector<double> &weights, double bias){
    double sum = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        sum += *(inputs[i]) * weights[i];
    }
    sum += bias;
    output = 1 / (1 + exp(-sum));
}

/**
 * ReLU activation function
 *
 * @param output   Output of the perceptron
 * @param inputs   Inputs of the perceptron
 * @param weights  Weights of the perceptron
 * @param bias     Bias of the perceptron
 */
void ReLU(double &output, std::vector<double *> &inputs, std::vector<double> &weights, double bias){
    double sum = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        sum += *(inputs[i]) * weights[i];
    }
    sum += bias;
    output = std::max(0.0, sum);
}

/**
 * Softmax activation function. The function gets the weighted sum of each perceptron in the layer and returns the
 * softmax of each perceptron
 *
 * @param weighted_sums   The weighted sums of all the perceptrons in the layer
 * @param index           The index of the current perceptron in the layer
 *
 * @return                Output of the perceptron
 */
std::vector<double> Softmax(std::vector<double *> &weighted_sums){
    std::vector<double> outputs;
    double sum = 0;
    for (auto & weighted_sum : weighted_sums) {
        sum += exp(*weighted_sum);
    }

    for (auto & weighted_sum : weighted_sums) {
        outputs.push_back(exp(*weighted_sum) / sum);
    }
    return outputs;
}

double SigmoidDerivative(double output) {
    return output * (1 - output);
}

double ReLUDerivative(double output) {
    return output > 0 ? 1 : 0;
}

double SoftmaxDerivative(double output) {
    return output * (1 - output);
}