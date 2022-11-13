#include <iostream>
#include <cstring>

#include "mnist/MNIST_Import.h"
#include "utils/Timer.h"
#include "ncc/NCC.h"
#include "../include/progressbar.h"


/**
 * Main function classifies the test images using the Nearest Centroid Classification algorithm. The arguments are:
 *   - The path to the dataset directory. The directory should contain the files:
 *     - train-images-idx3-ubyte
 *     - train-labels-idx1-ubyte
 *     - t10k-images-idx3-ubyte
 *     - t10k-labels-idx1-ubyte
 *
 *   Optional arguments:
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
    if (argc < 3){
        std::cerr << "Usage: " << argv[0]
                  << " -d <dataset directory> -k <value of K> [-n <number of test images>"
                     " -s <starting index for tests>]"
                  << std::endl;
    }

    std::string dataset_dir = argv[2];

    int n_tests = -1;
    int start_index = -1;

    for (int i = 3; i < argc - 1; i+=2) {
        if (strcmp(argv[i], "-n") == 0){
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

    n_tests = n_tests == -1 ? 10000 : n_tests;
    start_index = start_index == -1 ? 0 : start_index;

    if (start_index + n_tests > 10000){
        std::cerr << "The starting index + number of test images must be less than 10000" << std::endl;
        return 1;
    }

    std::cout << "Arguments: " << std::endl << std::endl;
    std::cout << "    Dataset directory: " << dataset_dir << std::endl;
    std::cout << "    Number of test images: " << n_tests << std::endl;
    std::cout << "    Starting index: " << start_index << std::endl;
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
    std::cout << std::endl << "Importing the images and labels..." << std::endl << std::endl;

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
    std::cout << "    Time to read the data: ";
    timer.displayElapsed();

    /*
     * Save the first image from the training set and the first image from the test set as pgm files to confirm that
     * the data was read correctly
     */

    std::cout << std::endl << "Saving training image as 'train_0.pgm'... Label: " << (int)training_images.at(0)->getLabel() << std::endl;
    training_images.at(0)->saveImage("images/train_0");

    std::cout << "Saving test image as 'test_0.pgm'... Label: " << (int)test_images.at(0)->getLabel() << std::endl << std::endl;
    test_images.at(0)->saveImage("images/test_0");

    std::cout << std::endl;

    // Start the classification process
    timer.startTimer();

    // Create the NCC object
    NCC ncc(training_images, test_images);

    std::cout << "Determine the mean vector for each class..." << std::endl;
    std::cout << std::endl;
    ncc.calculateMeans();  // Calculate the means for each class

    std::cout << std::endl;

    timer.stopTimer();
    std::cout << "    Time to calculate the means: ";
    timer.displayElapsed();

    // save the means as pgm files
    for (int i = 0; i < 10; i++){
        ncc.getClassMeans().at(i)->saveImage("images/mean_" + std::to_string(i));
    }

    std::cout << std::endl;
    std::cout << "Starting the classification..." << std::endl;
    std::cout << std::endl;

    timer.startTimer();

    progressbar bar(n_tests);

    std::cout << "    Classifying test images ";

    int miss = 0;  // The number of miss images

    for (int i = 0; i < n_tests; ++i) {
        bar.update();
        int res = ncc.classifyImage(i + start_index, false);  // Classify the image

        // If the result is not the same as the label, save the image
        if (res != test_images.at(i + start_index)->getLabel() && miss < 10){
            miss++;
            test_images.at(i + start_index)->saveImage(
                    "images/ncc_misclassified/miss_" + std::to_string(miss) + "_res_" + std::to_string(res) +
                    "_label_" + std::to_string(test_images.at(i + start_index)->getLabel())
            );
        }
    }

    std::cout << std::endl;
    std::cout << std::endl;

    timer.stopTimer();
    std::cout << "    Time to classify the images: ";
    timer.displayElapsed();
    std::cout << std::endl;

    std::cout << "Classification Summary:" << std::endl << std::endl;
    ncc.printStats();  // Print the statistics

    return 0;
}