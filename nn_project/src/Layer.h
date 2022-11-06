#ifndef NN_PROJECT_LAYER_H
#define NN_PROJECT_LAYER_H

#include "Perceptron.h"

/**
 * @brief A class that represents a layer of perceptrons. For this assignment we will only use fully connected layers.
 */
class Layer {
public:
    // Default constructor/destructor
    Layer()= default;
    ~Layer()= default;

private:
    Perceptron *perceptrons;  /// The perceptrons of the layer
    int n_perceptrons;        /// The number of perceptrons in the layer
};


#endif //NN_PROJECT_LAYER_H
