#ifndef NN_PROJECT_PERCEPTRON_H
#define NN_PROJECT_PERCEPTRON_H

#include <vector>
#include <array>

class Perceptron {
public:
    // Constructors
    Perceptron()= delete;
    explicit Perceptron(int n_inputs, bool input_layer=false, bool output_layer=false);

    // Destructor
    ~Perceptron()= default;

    // Getters
    double getWeight(int index) const;
    double getError() const;
    double *getOutputPtr();
    int getNInputs() const;

    // Setters
    void setWeights(std::vector<double>& set_weights);
    void setWeight(double weight, int index);
    void setInputs(std::vector<double *>& set_inputs);
    void setInput(double* input, int index);
    void setOutput(double set_output);
    void setBias(double set_bias);
    void setActivationFunction(void (*activation_function)(double &, std::vector<double *> &, std::vector<double> &, double));
    void setActivationFunctionDerivative(double (*activation_function_derivative)(double));

    // Functions
    void activate();
    void updateError(double target = 0, const std::vector<double>& next_layer_errors = {},
                     const std::vector<double>& next_layer_weights = {});
    void updateGradient();
    void changeWeightsAndBias(double learning_rate);
    void printPerceptron() const;

private:
    std::vector<double *> inputs {};   // Inputs of the perceptron (pointers to the output of the previous layer)
    std::vector<double> weights {};    // Weights of the perceptron
    double output {0};                 // Output of the perceptron
    double bias {0};                   // Bias of the perceptron

    int n_samples {0};                 // Number of samples used to calculate the average output
    double error {0};                  // Sum of the errors used to calculate the average error
    bool is_input {false};             // True if the perceptron is an input perceptron
    bool is_output {false};            // True if the perceptron is an output perceptron
    double bias_grad_sum {0};          // Sum of the bias gradients used to calculate the average bias gradient
    std::vector<double> weight_grad_sum {}; // Sum of the weight gradients used to calculate the average weight gradient

    // Pointers to the activation function and its derivative
    void (*activationFunction)(double &, std::vector<double *> &, std::vector<double> &, double) {nullptr};
    double (*activationFunctionDerivative)(double) {nullptr};
};

#endif