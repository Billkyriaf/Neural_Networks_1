#ifndef NN_PROJECT_PERCEPTRON_H
#define NN_PROJECT_PERCEPTRON_H

class Perceptron {
public:
    // Default constructor/destructor
    Perceptron()= default;
    ~Perceptron()= default;

    double (*activationFunction)(double);  /// activation function pointer

    // Functions
    void initializeWeights(const double* initial_weights);
    double activate(const double *input);

private:
    double *weights;  /// The weights of the perceptron (one for each input)
    int n_inputs;     /// The number of inputs to the perceptron
    double *output;   /// The output of the perceptron
};

/**
 * @brief Pass the initial values to the weights array
 * @param initial_weights  The initial values of the weights
 * @param size  The size of the weights array
 */
void Perceptron::initializeWeights(const double* initial_weights) {
    for (int i = 0; i < this->n_inputs; ++i) {
        this->weights[i] = initial_weights[i];
    }
}

/**
 * Sums all the inputs multiplied by their respective weights and calls the activation function
 * @param input  The array of inputs
 * @return  The result of the activation function
 */
double Perceptron::activate(const double *input) {
    double sum = 0;

    // Calculate the sum of the inputs multiplied by their respective weights
    for (int i = 0; i < sizeof(input); i++) {
        sum += input[i] * this->weights[i];
    }

    // Update the output of the perceptron
    *(this->output) =  this->activationFunction(sum);

    // Return the output of the perceptron
    return *(this->output);
}

#endif