<div id="top"></div>

<br />
<div align="center">
  <h1 align="center">Neural Network MNIST Classifier</h1>
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

- [1. About This Project](#1-about-this-project)
- [2. Getting Started](#2-getting-started)
- [3. Dependencies](#3-dependencies)
    - [3.1. Make](#31-make)
    - [3.2. C++ 14 capable Compiler](#32-c-14-capable-compiler)
    - [3.3. progressbar Library](#33-progressbar-library)
    - [3.4. Mnist Dataset](#34-mnist-dataset)
- [4. Usage](#4-usage)
    - [4.1. `make` targets](#41-make-targets)
      - [nn classifier](#nn-classifier)


## 1. About This Project

The main point of focus of this project is to create a Neural Network and train it on a dataset. For the purposes of this assignment the MNIST data set was used.


## 2. Getting Started

To setup this repository on your local machine run the following command on the terminal:

```console
$ git clone git@github.com:Billkyriaf/Neural_Networks_1.git
```

Or alternatively [*download*](https://github.com/Billkyriaf/Neural_Networks_1/archive/refs/heads/main.zip) and extract the zip file of the repository.

## 3. Dependencies
#### 3.1. Make

This project uses make utilities to build and run the executables.

#### 3.2. C++ 14 capable Compiler

This project uses C++ 14 features and requires a compiler that supports C++ 14.

#### 3.3. progressbar Library

This project uses the ***progressbar*** library by **gipert** to display a progress bar during the execution of the program. The library is included in the repository and no additional setup is required. For more information about the library visit the [*github page*](https://github.com/gipert/progressbar).

#### 3.4. Mnist Dataset

The dataset used in this project is the Mnist dataset. The dataset is included in the repository and no additional setup is required.

## 4. Usage

To build the executables from the root directory of the repository run the following command on the terminal:

```console
$ cd nn_project
```

In the nn_project directory you can find the Makefile that is used to build the executable.

`IMPORTANT!` Before building the executables make sure that the MNIST dataset is in the `data` directory. If the dataset is not in the `data` directory you can download it from the [*official website*](http://yann.lecun.com/exdb/mnist/) and extract it in the `data` directory. The `data` directory should contain the following files: 
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

#### 4.1. `make` targets

The Makefile contains the following targets:

##### nn classifier

```console
# nn classifier
$ make run_nn  
```
Arguments:

```console
# The nn executable requires no arguments
```

To change the default parameters of the NN edit the main.cpp [here](https://github.com/Billkyriaf/Neural_Networks_1/blob/bfac419b352efc1cd2c4d8220ac97e489add608f/nn_project/src/main.cpp#L36).