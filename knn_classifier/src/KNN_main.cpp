#include <iostream>
#include <cmath>
#include <cstring>

#include "mnist/MNIST_Import.h"
#include "knn/KNN.h"
#include "utils/Timer.h"
#include "utils/Print_Progress.h"


/**
 * Struct with the arguments for the threads
 */
typedef struct {
    int thread_id;   /// The thread ID
    int start;       /// The index of the first image to process
    int end;         /// The index of the last image to process

    pthread_mutex_t *mutex; /// The mutex to lock the progress

    Print_Progress *progress;  /// The progress object
    KNN *knn;                  /// The KNN object
} thread_data;


/**
 * Thread function for each classifier. Classifies the images in the range [start, end)
 *
 * @param arg The thread arguments
 * @return nullptr
 */
void *classify(void *arg) {
    // Type cast the arguments
    auto *data = (thread_data *) arg;

    // Classify the images
    for (int i = data->start; i < data->end; i++) {
        data->knn->classifyImage(i, false);

        // Update the progress bar
        if ((data->end - data->start) / 8 > 0) {
            if (i % ((data->end - data->start) / 8) == 0 && i != 0) {
                // Lock the mutex
                pthread_mutex_lock(data->mutex);

                // Update the progress bar
                data->progress->setProgress(data->thread_id, i - data->start);

                // Unlock the mutex
                pthread_mutex_unlock(data->mutex);
            }
        }
    }

    // Update the progress bar to 100%
    pthread_mutex_lock(data->mutex);  // Lock the mutex
    data->progress->setProgress(data->thread_id, data->end - data->start);  // Update the progress bar
    pthread_mutex_unlock(data->mutex);  // Unlock the mutex

    pthread_exit(nullptr);
}

/**
 * Main function classifies the test images using the KNN algorithm. The arguments are:
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
int main(int argc, char *argv[]) {
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


    std::vector<KNN *> classifiers;  // Create a KNN classifiers
    classifiers.reserve(n_threads);

    for (int i = 0; i < n_threads; i++) {
        classifiers.push_back(new KNN(k, training_images, test_images));
    }

    // Create the threads. Every classifier will be run in a separate thread and will classify a part of the test images
    pthread_t threads[n_threads];
    thread_data data[n_threads];

    // The mutex is used to lock the progress bar
    auto *mutex = new pthread_mutex_t;
    pthread_mutex_init(mutex, nullptr);

    // The progress bar object
    auto *progress = new Print_Progress(n_threads, int(n_tests / n_threads));

    // Create the threads
    for (int i = 0; i < n_threads - 1; i++) {
        data[i].thread_id = i;
        data[i].start = i * int(n_tests / n_threads) + start_index;
        data[i].end = (i + 1) * int(n_tests / n_threads) + start_index;
        data[i].knn = classifiers.at(i);
        data[i].progress = progress;
        data[i].mutex = mutex;

        pthread_create(&threads[i], nullptr, classify, &data[i]);
    }

    // The last thread will classify the remaining images
    if (n_threads >= 1) {
        data[n_threads - 1].thread_id = n_threads - 1;
        data[n_threads - 1].start = (n_threads - 1) * int(n_tests / n_threads) + start_index;
        data[n_threads - 1].end = n_tests + start_index;
        data[n_threads - 1].knn = classifiers.at(n_threads - 1);
        data[n_threads - 1].progress = progress;
        data[n_threads - 1].mutex = mutex;

        pthread_create(&threads[n_threads - 1], nullptr, classify, &data[n_threads - 1]);
    }

    // Wait for the threads to finish
    for (unsigned long thread : threads) {
        pthread_join(thread, nullptr);
    }

    timer.stopTimer();
    std::cout << std::endl << std::endl  << "Time to classify all the test images: ";
    timer.displayElapsed();

    classifiers.at(0)->accumulateStats(classifiers);
    classifiers.at(0)->printStats();

    // Free the memory
    for (uint32_t i = 0; i < mnist.getTrDataCount(); ++i) {
        delete training_images.at(i);
    }

    for (uint32_t i = 0; i < mnist.getTsDataCount(); ++i) {
        delete test_images.at(i);
    }

    pthread_mutex_destroy(mutex);
    delete(progress);
    delete(mutex);

    return 0;
}
