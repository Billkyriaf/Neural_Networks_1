#include <iostream>
#include <cstring>

#include "mnist/MNIST_Import.h"
#include "utils/Timer.h"
#include "ncc/NCC.h"


/**
 * Main function classifies the test images using the Nearest Centroid Classification algorithm. The arguments are:
 *   - The path to the dataset directory. The directory should contain the files:
 *     - train-images-idx3-ubyte
 *     - train-labels-idx1-ubyte
 *     - t10k-images-idx3-ubyte
 *     - t10k-labels-idx1-ubyte
 *
 *   - The value of K
 *
 *   Optional arguments:
 *   - The number of threads to use
 *   - The number of test images to classify
 *   - The starting index of the test images
 *
 * ./main -d /home/username/dataset -k 5 -t 16 -n 10000 -s 0
 *
 *
 * @return 0
 */
int main(int argc, char *argv[]){
    // Parse the arguments
    if (argc < 5){
        std::cerr << "Usage: " << argv[0]
                  << " -d <dataset directory> -k <value of K> [-t <number of threads> -n <number of test images>"
                     " -s <starting index for tests>]"
                  << std::endl;
    }

    std::string dataset_dir = argv[2];
    int k = std::stoi(argv[4]);

    int n_threads = -1;
    int n_tests = -1;
    int start_index = -1;

    for (int i = 5; i < argc - 1; i+=2) {
        if (strcmp(argv[i], "-t") == 0){
            n_threads = std::stoi(argv[i + 1]);

            if (n_threads < 1 || n_threads > 16){
                std::cerr << "The number of threads must be greater than 0 and less than 256" << std::endl;
                return 1;
            }

        } else if (strcmp(argv[i], "-n") == 0){
            n_tests = std::stoi(argv[i + 1]);

            if (n_tests < 1 || n_tests > 10000){
                std::cerr << "The number of test images must be greater than 0 and less than 10000" << std::endl;
                return 1;
            }

        } else if (strcmp(argv[i], "-s") == 0){
            start_index = std::stoi(argv[i + 1]);

            if (start_index < 0 || start_index > 10000){
                std::cerr << "The starting index must be greater/equal than 0 and less than 10000" << std::endl;
                return 1;
            }

        } else {
            std::cerr << "Invalid argument: " << argv[i] << std::endl;
            return 1;
        }

    }

    n_threads = n_threads == -1 ? 16 : n_threads;
    n_tests = n_tests == -1 ? 10000 : n_tests;
    start_index = start_index == -1 ? 0 : start_index;

    if (start_index + n_tests > 10000){
        std::cerr << "The starting index + number of test images must be less than 10000" << std::endl;
        return 1;
    }

    if (n_tests < n_threads){
        n_threads = n_tests;
    }

    std::cout << "Dataset directory: " << dataset_dir << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "Number of threads: " << n_threads << std::endl;
    std::cout << "Number of test images: " << n_tests << std::endl;
    std::cout << "Starting index: " << start_index << std::endl;
    std::cout << std::endl;




    Timer timer;  // The timer object is used to time the program
    timer.startTimer();  // Start the timer

    // Create the import object
    MNIST_Import mnist(
            dataset_dir + "/train-images.idx3-ubyte",
            dataset_dir + "/train-labels.idx1-ubyte",
            dataset_dir + "/t10k-images.idx3-ubyte",
            dataset_dir + "/t10k-labels.idx1-ubyte"
    );

    //Start importing the images and labels

    mnist.readMetadata();  // Read the metadata from the file
    mnist.printMetadata();  // Print the metadata


    std::vector<MNIST_Image *> training_images;  // The vector to store the training images
    training_images.reserve(mnist.getTrDataCount());  // Reserve the memory for better performance

    mnist.readTrainingData(training_images);  // Read the training data


    std::vector<MNIST_Image *> test_images;  // The vector to store the test images
    test_images.reserve(mnist.getTsDataCount());  // Reserve the memory for better performance

    mnist.readTestData(test_images);  // Read the test data

    // Elapsed time for importing the images
    timer.stopTimer();
    std::cout << "Time to read the data: ";
    timer.displayElapsed();

    /*
     * Save the first image from the training set and the first image from the test set as pgm files to confirm that
     * the data was read correctly
     */

    std::cout << "Saving training image as 'train_0.pgm'... Label: " << (int)training_images.at(0)->getLabel() << std::endl;
    training_images.at(0)->saveImage("images/train_0");

    std::cout << "Saving test image as 'test_1.pgm'... Label: " << (int)test_images.at(0)->getLabel() << std::endl;
    test_images.at(0)->saveImage("images/test_0");

    std::cout << std::endl;
    std::cout << std::endl;


    // Start the classification process
    timer.startTimer();

    // Create the NCC object
    NCC ncm(training_images, test_images);

    ncm.calculateMeans();  // Calculate the means for each class

    // save the means as pgm files
    for (int i = 0; i < 10; i++){
        ncm.getClassMeans().at(i)->saveImage("images/mean_" + std::to_string(i));
    }

    for (int i = 0; i < n_tests; ++i) {
        ncm.classifyImage(i + start_index, false);  // Classify the image
    }

    ncm.printStats();  // Print the statistics

    timer.stopTimer();
    std::cout << "Calculating the means took: ";
    timer.displayElapsed();

    return 0;
}