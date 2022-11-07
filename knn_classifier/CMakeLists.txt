cmake_minimum_required(VERSION 3.23)
project(knn_classifier)

set(CMAKE_CXX_STANDARD 14)

add_executable(knn_classifier src/main.cpp src/MNIST_Image.cpp src/MNIST_Image.h)
