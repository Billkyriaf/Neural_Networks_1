cmake_minimum_required(VERSION 3.23)
project(nn_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(nn_project src/main.cpp src/perceptrons/Perceptron.h src/perceptrons/Perceptron.cpp src/Network.cpp src/Network.h
        src/mnist/MNIST_Import.cpp src/mnist/MNIST_Import.h src/mnist/MNIST_Image.cpp src/mnist/MNIST_Image.h src/network_functions/activation_functions.cpp src/network_functions/activation_functions.h src/network_functions/initialization_functions.cpp src/network_functions/initialization_functions.h)
