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

double SigmoidDerivative(double output) {
    return output * (1 - output);
}

double ReLUDerivative(double output) {
    return output > 0 ? 1 : 0;
}