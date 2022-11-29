#ifndef NN_PROJECT_ACTIVATION_FUNCTIONS_H
#define NN_PROJECT_ACTIVATION_FUNCTIONS_H

#include <vector>

// TODO : Add softmax function

// Activation functions
void Sigmoid(double &output, std::vector<double *> &inputs, std::vector<double> &weights, double bias);

void ReLU(double &output, std::vector<double *> &inputs, std::vector<double> &weights, double bias);

std::vector<double> Softmax(std::vector<double *> &weighted_sums);

double SigmoidDerivative(double output);

double ReLUDerivative(double output);

double SoftmaxDerivative(double output);
#endif