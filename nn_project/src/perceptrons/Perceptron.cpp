#include <iostream>
#include "Perceptron.h"

/**
 * Constructor of the Perceptron class
 * @param n_inputs      Number of inputs of the perceptron
 * @param input_layer   True if the perceptron is an input perceptron
 * @param output_layer  True if the perceptron is an output perceptron
 */
Perceptron::Perceptron(int n_inputs,  bool input_layer,  bool output_layer) : is_input(input_layer), is_output(output_layer) {
    // Initialize weights and inputs
    Perceptron::weights.reserve(n_inputs);
    Perceptron::inputs.reserve(n_inputs);
    Perceptron::weight_grad_sum.reserve(n_inputs);

    // Initialize weights and inputs
    for (int i = 0; i < n_inputs; ++i) {
        Perceptron::weights.push_back(0);
        Perceptron::inputs.push_back(nullptr);
        Perceptron::weight_grad_sum.push_back(0);
    }
}


// ======================= Getters =======================

double Perceptron::getWeight(int index) const {
    return Perceptron::weights[index];
}

double Perceptron::getError() const {
    return Perceptron::error;
}

double *Perceptron::getOutputPtr() {
    return &(Perceptron::output);
}

int Perceptron::getNInputs() const {
    return int(Perceptron::weights.size());
}


// ======================= Setters =======================

void Perceptron::setWeights(std::vector<double>& set_weights) {
    Perceptron::weights = set_weights;
}

void Perceptron::setWeight(double weight, int index) {
    Perceptron::weights.at(index) = weight;
}

void Perceptron::setInputs(std::vector<double *> &set_inputs) {
    Perceptron::inputs = set_inputs;
}

void Perceptron::setInput(double *input, int index) {
    Perceptron::inputs.at(index) = input;
}

void Perceptron::setOutput(double set_output) {
    Perceptron::output = set_output;
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


// ======================= Functions =======================

/**
 * The forward propagation function of the perceptron. If the perceptron is an input perceptron, the output is simply the
 * input. Otherwise, the activation function is called.
 */
void Perceptron::activate() {

    if (Perceptron::is_input){
        // If the perceptron is an input perceptron, the output is simply the input
        Perceptron::output = *(Perceptron::inputs.at(0));

    } else if (Perceptron::is_output){
        // If the perceptron is an output perceptron, the output is the weighted sum of the inputs and the bias. The
        // activation will be called from the network class
        double sum = 0;
        for (int i = 0; i < Perceptron::inputs.size(); ++i) {
            sum += *(Perceptron::inputs[i]) * Perceptron::weights[i];
        }

        Perceptron::output = sum + Perceptron::bias;

//        Perceptron::activationFunction(Perceptron::output, Perceptron::inputs, Perceptron::weights, Perceptron::bias);


    } else if (Perceptron::activationFunction != nullptr) {
        // else call the activation function
        Perceptron::activationFunction(Perceptron::output, Perceptron::inputs, Perceptron::weights, Perceptron::bias);

    } else {
        std::cerr << "Error: activation function not set" << std::endl;
    }
}


/**
 * The error is calculated for every training sample. This error is a combination of some of the partial derivatives
 * therms of the cost function.
 *
 * For the output layer the error is:
 *
 *    δ = (y_hat - y) * f'(y)
 *
 *    where:
 *        y is the output of the perceptron
 *        y_hat is the target value
 *        f' is the derivative of the activation function with respect to the output of the perceptron
 *
 * For every other layer the error is:
 *
 *    δ = (Σ(wi * δi_next)) * f'(y)
 *
 *    where:
 *        wi is the weight of the connection between the current perceptron and the next perceptron
 *        δi_next is the error of the next layer perceptron
 *        f' is the derivative of the activation function with respect to the output of the perceptron
 *
 *
 * @param target              The target value of the output layer
 * @param next_layer_errors   The errors of the next layer perceptrons
 * @param next_layer_weights  The weights of the next layer perceptrons
 */
void Perceptron::updateError(double target, const std::vector<double>& next_layer_errors,
                             const std::vector<double>& next_layer_weights) {

    if (Perceptron::is_output) {
        // if the perceptron is an output perceptron, the error is calculated as described above
        Perceptron::error = (Perceptron::output - target) * Perceptron::activationFunctionDerivative(Perceptron::output);

    } else {
        // else the error is calculated as the sum of the errors of the next layer perceptrons weighted by the weights
        double error_sum = 0;
        for (int i = 0; i < next_layer_errors.size(); ++i) {
            error_sum += next_layer_weights[i] * next_layer_errors[i] * Perceptron::activationFunctionDerivative(Perceptron::output);
        }

        Perceptron::error = error_sum;
    }

    // update the number of samples
    Perceptron::n_samples++;
}


/**
 * The weight gradient is averaged for every training sample. In this function only the sum of the average is
 * calculated and the actual average is calculated in the changeWeightsAndBias function. Same applies for the bias
 *
 * For the weights the gradient of the cost function is:
 *
 *    ∂C/∂wi = δ * yi
 *
 *    where:
 *        δ is the error of the perceptron
 *        yi is the input of the perceptron
 *
 * For the bias the gradient of the cost function is:
 *
 *    ∂C/∂b = δ
 *
 *    where:
 *        δ is the error of the perceptron
 *
 */
void Perceptron::updateGradient() {
    // update the gradient sum for all the weights of the perceptron
    for (int i = 0; i < Perceptron::weights.size(); ++i) {
        Perceptron::weight_grad_sum[i] += Perceptron::error * (*Perceptron::inputs[i]);
    }

    // update the gradient sum for the bias of the perceptron
    Perceptron::bias_grad_sum += Perceptron::error;
}

/**
 * The weights and bias are updated using the gradient descent algorithm. At this point the average of the gradients
 * is calculated.
 *
 * @param learning_rate   The learning rate of the gradient descent algorithm
 */
void Perceptron::changeWeightsAndBias(double learning_rate) {

    // update the weights
    for (int i = 0; i < Perceptron::weights.size(); ++i) {
        double weight_grad = Perceptron::weight_grad_sum[i] / Perceptron::n_samples;
        Perceptron::weights[i] -= learning_rate * weight_grad;
    }

    // update the bias
    Perceptron::bias -= learning_rate * (Perceptron::bias_grad_sum / Perceptron::n_samples);

    // reset the gradient sums
    Perceptron::weight_grad_sum = std::vector<double>(Perceptron::weights.size(), 0);
    Perceptron::bias_grad_sum = 0;
    Perceptron::n_samples = 0;
    Perceptron::error = 0;
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
