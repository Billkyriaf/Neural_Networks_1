<div id="top"></div>

<br />
<div align="center">
  <h1 align="center">KNN and K-means Classification</h1>
  <h3 align="center">Aristotle University of Thessaloniki</h3>
  <h4 align="center">School of Electrical & Computer Engineering</h4>
  <p align="center">
    Author: Kyriafinis Vasilis
    <br />
    Winter Semester 2022 - 2023
    <br />
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
- [1. About This Project](#1-about-this-project)
- [2. Getting Started](#2-getting-started)
- [3. Dependencies](#3-dependencies)
    - [3.1. Make](#31-make)
    - [3.2. C++ 11 capable Compiler](#32-c-11-capable-compiler)
    - [3.3. progressbar Library](#33-progressbar-library)
    - [3.4. Mnist Dataset](#34-mnist-dataset)
- [4. Usage](#4-usage)
    - [4.1. `make` targets](#41-make-targets)
      - [Knn Classifier](#knn-classifier)
      - [K-means Classifier](#k-means-classifier)
      - [K-means Clustering Classifier](#k-means-clustering-classifier)

## 1. About This Project

The goal of this project is to implement the KNN and K-means classification algorithms and compare their performance. The project is implemented in C++ and the results of the comparison can be found in the report pdf file.

## 2. Getting Started

To setup this repository on your local machine run the following command on the terminal:

```console
$ git clone git@github.com:Billkyriaf/Neural_Networks_1.git
```

Or alternatively [*download*](https://github.com/Billkyriaf/Neural_Networks_1/archive/refs/heads/main.zip) and extract the zip file of the repository.

## 3. Dependencies
#### 3.1. Make

This project uses make utilities to build and run the executables.

#### 3.2. C++ 11 capable Compiler

This project uses C++ 14 features and requires a compiler that supports C++ 14.

#### 3.3. progressbar Library

This project uses the ***progressbar*** library by **gipert** to display a progress bar during the execution of the program. The library is included in the repository and no additional setup is required. For more information about the library visit the [*github page*](https://github.com/gipert/progressbar).

#### 3.4. Mnist Dataset

The dataset used in this project is the Mnist dataset. The dataset is included in the repository and no additional setup is required.

## 4. Usage

To build the executables from the root directory of the repository run the following command on the terminal:

```console
$ cd knn_classifier
```

In the knn_classifier directory you can find the Makefile that is used to build the executables.

`IMPORTANT!` Before building the executables make sure that the MNIST dataset is in the `data` directory. If the dataset is not in the `data` directory you can download it from the [*official website*](http://yann.lecun.com/exdb/mnist/) and extract it in the `data` directory. The `data` directory should contain the following files: 
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

#### 4.1. `make` targets

The Makefile contains the following targets:

##### Knn Classifier

```console
# knn classifier
$ make run_knn  
```
Arguments:

```console
# The kkn executable requires 2 mandatory arguments and 3 optional arguments
# Mandatory arguments:
#   -d <dataset>  : The directory of the dataset
#   -k <int>      : The number of neighbors to use
#
# Optional arguments:
#   -t <int>  : The number of threads to use (default: 16)
#   -n <int>  : The number of images to use for testing (default: 10000)
#   -s <int>  : The starting index of the testing images (default: 0)
```

To change the arguments edit the Makefile [here](https://github.com/Billkyriaf/Neural_Networks_1/blob/39fde23404f6caea81df83d3e2f089cc17091f5a/knn_classifier/Makefile#L75).

##### K-means Classifier


```console
# k-means classifier
$ make run_ncc
```
Arguments:

```console
# The ncc executable requires 1 mandatory arguments and 2 optional arguments
# Mandatory arguments:
#   -d <dataset>  : The directory of the dataset
#
# Optional arguments:
#   -n <int>  : The number of images to use for testing (default: 10000)
#   -s <int>  : The starting index of the testing images (default: 0)
```

To change the arguments edit the Makefile [here](https://github.com/Billkyriaf/Neural_Networks_1/blob/39fde23404f6caea81df83d3e2f089cc17091f5a/knn_classifier/Makefile#L85).

##### K-means Clustering Classifier

```console
# k-means clustering
$ make run_ncc_clustering
```

Arguments:

```console
# The ncc_clustering executable requires 2 mandatory arguments and 1 optional flag
# Mandatory arguments:
#   -d <dataset>  : The directory of the dataset
#   -c <int>      : The number of clusters to use
#
# Optional arguments:
#   -fit  : If the fit flag is set the program will fit the clusters from scratch else it will use
#           the clusters from the previous runs. The first time the program must be run with the fit
#           flag set. Every time the clusters are incremented the fit flag must be set.
```
To change the arguments edit the Makefile [here](https://github.com/Billkyriaf/Neural_Networks_1/blob/39fde23404f6caea81df83d3e2f089cc17091f5a/knn_classifier/Makefile#L95).
