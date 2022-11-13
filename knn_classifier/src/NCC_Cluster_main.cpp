#include <iostream>
#include <cstring>

#include "mnist/MNIST_Import.h"
#include "utils/Timer.h"
#include "ncc_cluster/NCC_clusters.h"
#include "../include/progressbar.h"

/**
 * Main function classifies the test images using the Nearest Centroid Classification optimized with clusters.
 *
 * The arguments are:
 *   -d The path to the dataset directory. The directory should contain the files:
 *     * train-images-idx3-ubyte
 *     * train-labels-idx1-ubyte
 *     * t10k-images-idx3-ubyte
 *     * t10k-labels-idx1-ubyte
 *
 *   -c The number of clusters
 *
 *   Optional arguments:
 *   -fit Whether to train the clusters from scratch or use the pre-trained clusters
 *
 * ./main -d /home/username/dataset -c 5 -t 16 -n 10000 -s 0
 *
 *
 * @return 0
 */
int main(int argc, char *argv[]){
    // Parse the arguments
    if (argc < 5){
        std::cerr << "Usage: " << argv[0] << " -d <dataset directory> -c <The number of clusters> [-fit ]" << std::endl;
    }

    std::string dataset_dir = argv[2];
    int n_clusters = std::stoi(argv[4]);

    bool from_scratch = false;

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "-fit") == 0){
            from_scratch = true;
        } else {
            std::cerr << "Invalid argument: " << argv[i] << std::endl;
            return 1;
        }

    }

    std::cout << "Arguments:" << std::endl << std::endl;
    std::cout << "    Dataset directory: " << dataset_dir << std::endl;
    std::cout << "    Number of clusters: " << n_clusters << std::endl;
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
    std::cout << std::endl << "Importing images and labels..." << std::endl << std::endl;
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

    std::cout << "Saving test image as 'test_1.pgm'... Label: " << (int)test_images.at(0)->getLabel() << std::endl;
    test_images.at(0)->saveImage("images/test_0");

    std::cout << std::endl;
    std::cout << std::endl;


    // Start the classification process
    timer.startTimer();

    // Create the classifier object
    if (from_scratch){
        std::cout << "Creating the clusters from scratch..." << std::endl;

        NCC_clusters ncc_cluster(n_clusters, training_images, test_images);
        ncc_cluster.saveMeanClusters();  // Save the mean clusters as pgm files

        timer.stopTimer();
        std::cout << "    Time to create and fit the classifier: ";
        timer.displayElapsed();

        // Start the classification process
        timer.startTimer();

        std::cout << std::endl << "Starting the classification..." << std::endl << std::endl;

        progressbar bar(int(test_images.size()));  // Create the progress bar

        std::cout << "    Classifying the test images  ";
        for (int i = 0; i < test_images.size(); ++i) {
            bar.update();
            ncc_cluster.classifyImage(i, false);
        }
        std::cout << std::endl;
        timer.stopTimer();
        std::cout << std::endl << "    Time to classify the test images: ";
        timer.displayElapsed();

        ncc_cluster.printStats();

    } else {
        std::cout << "Loading the clusters from file..." << std::endl;

        NCC_clusters ncc_cluster("pre_fit", training_images, test_images);

        timer.stopTimer();
        std::cout << std::endl << "    Time to create the classifier from the pre-saved mean clusters: ";
        timer.displayElapsed();

        // Start the classification process
        timer.startTimer();

        std::cout << "Starting the classification..." << std::endl << std::endl;

        progressbar bar(int(test_images.size()));  // Create the progress bar

        std::cout << "    Classifying the test images  ";

        for (int i = 0; i < test_images.size(); ++i) {
            bar.update();
            ncc_cluster.classifyImage(i, false);
        }
        std::cout << std::endl;

        timer.stopTimer();
        std::cout << std::endl << "    Time to classify the 10000 test images: ";
        timer.displayElapsed();

        ncc_cluster.printStats();
    }

    return 0;
}