#ifndef NN_PROJECT_INITIALIZATION_FUNCTIONS_H
#define NN_PROJECT_INITIALIZATION_FUNCTIONS_H

#include "../perceptrons/Perceptron.h"

void ZeroInitialization(std::vector<std::vector<Perceptron *>> &network);

void XavierInitialization(std::vector<std::vector<Perceptron *>> &network);

void NormalizedXavierInitialization(std::vector<std::vector<Perceptron *>> &network);

void KaimingInitialization(std::vector<std::vector<Perceptron *>> &network);

#endif
