cmake_minimum_required(VERSION 3.23)
project(knn_classifier)

set(CMAKE_CXX_STANDARD 14)

# Set -O3 optimization flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(knn_classifier src/KNN_main.cpp src/mnist/MNIST_Image.cpp src/mnist/MNIST_Image.h
        src/mnist/MNIST_Import.cpp src/mnist/MNIST_Import.h src/knn/KNN.cpp src/knn/KNN.h src/utils/Timer.cpp
        src/utils/Timer.h src/utils/Print_Progress.cpp src/utils/Print_Progress.h include/progressbar.h)

add_executable(nc_classifier src/NCC_main.cpp src/ncc/NCC.cpp src/ncc/NCC.h src/mnist/MNIST_Image.cpp
        src/mnist/MNIST_Image.h src/mnist/MNIST_Import.cpp src/mnist/MNIST_Import.h src/utils/Timer.cpp src/utils/Timer.h include/progressbar.h)

add_executable(ncc_cluster src/NCC_Cluster_main.cpp src/ncc/NCC_clusters.cpp src/ncc/NCC_clusters.cpp src/mnist/MNIST_Image.cpp
        src/mnist/MNIST_Image.h src/mnist/MNIST_Import.cpp src/mnist/MNIST_Import.h src/utils/Timer.cpp src/utils/Timer.h include/progressbar.h)