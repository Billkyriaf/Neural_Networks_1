#ifndef NN_PROJECT_ACTIVATION_FUNCTIONS_H
#define NN_PROJECT_ACTIVATION_FUNCTIONS_H

#include <vector>

// Activation functions
void Sigmoid(double &output, std::vector<double *> &inputs, std::vector<double> &weights, double bias);

void ReLU(double &output, std::vector<double *> &inputs, std::vector<double> &weights, double bias);

// Activation functions derivatives with respect to the output of the perceptron
double SigmoidDerivative(double output);

double ReLUDerivative(double output);

#endif