cmake_minimum_required(VERSION 3.23)
project(knn_classifier)

set(CMAKE_CXX_STANDARD 14)

# Set -O3 optimization flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(knn_classifier src/main.cpp src/MNIST_Image.cpp src/MNIST_Image.h src/MNIST_Import.cpp src/MNIST_Import.h src/KNN.cpp src/KNN.h src/Timer.cpp src/Timer.h)
