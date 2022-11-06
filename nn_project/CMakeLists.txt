cmake_minimum_required(VERSION 3.23)
project(nn_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(nn_project src/main.cpp src/Perceptron.h src/Layer.h)
