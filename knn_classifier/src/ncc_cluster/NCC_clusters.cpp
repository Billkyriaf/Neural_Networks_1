#include <algorithm>
#include <limits>
#include <fstream>
#include <chrono>
#include <random>

#include "../../include/progressbar.h"
#include "NCC_clusters.h"

#define N_THREADS 16
#define N_ITERATIONS 30

/**
 * Constructor
 * @param n_clusters        The number of clusters
 * @param training_images   The training images
 * @param test_images       The test images
 */
NCC_clusters::NCC_clusters(int n_clusters, const std::vector<MNIST_Image *> &training_images,
                           const std::vector<MNIST_Image *> &test_images) : n_clusters(n_clusters), from_file(false) {
    // seed the random number generator
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = new std::default_random_engine(seed);

    // Deep copy the training images
    for (auto & training_image : training_images) {
        auto *image = new MNIST_Image(*training_image);

        NCC_clusters::training_images.push_back(image);
    }

    // Deep copy the test images
    for (auto & test_image : test_images) {
        auto *image = new MNIST_Image(*test_image);

        NCC_clusters::test_images.push_back(image);
    }

    // Initialize the mean and the converge vectors for the clusters
    NCC_clusters::cluster_means.reserve(n_clusters);

    /*
     * The expected number of images in each cluster is not known. To gain some efficiency, we will reserve space for
     * 60000 / n_clusters images per cluster.
     */
    NCC_clusters::clusters.reserve(n_clusters);

    for (int i = 0; i < n_clusters; i++) {
        NCC_clusters::clusters.emplace_back();
        NCC_clusters::clusters.at(i).reserve(training_images.size() / n_clusters);
    }

    std::cout << std::endl << "    Initializing the clusters  ";

    // Initialize the centroid means using k-means++
    NCC_clusters::initializeCentroids(30);

    std::cout << std::endl;

    std::cout << std::endl << "    Fitting the clusters       ";

    progressbar bar((N_ITERATIONS) * 60);

    // Train the clusters
    for (int i = 0; i < N_ITERATIONS - 1; ++i) {
        // Update the progress bar
        for (int j = 0; j < 60; ++j) {
            bar.update();
        }

        NCC_clusters::fitClusters(false, 5);
    }

    // Final iteration
    NCC_clusters::fitClusters(true, 5);

    // Update the progress bar
    for (int j = 0; j < 60; ++j) {
        bar.update();
    }


    std::cout << std::endl;
    std::cout << std::endl;

    // Determine the all the mean_cluster labels
    NCC_clusters::determineClusterLabel();
}


/**
 * Constructor to load the clusters from files instead of training them
 *
 * @param cluster_dir the directory containing the cluster files
 */
NCC_clusters::NCC_clusters(const std::string& cluster_dir, const std::vector<MNIST_Image *>& training_images,
                           const std::vector<MNIST_Image *>& test_images) : from_file(true) {

    // seed the random number generator
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = new std::default_random_engine(seed);

    // Deep copy the training images
    for (auto & training_image : training_images) {
        auto *image = new MNIST_Image(*training_image);

        NCC_clusters::training_images.push_back(image);
    }

    // Deep copy the test images
    for (auto & test_image : test_images) {
        auto *image = new MNIST_Image(*test_image);

        NCC_clusters::test_images.push_back(image);
    }

    // open the stats file
    std::ifstream stats_file(cluster_dir + "/stats");

    // read the number of clusters
    int clusters_num;
    stats_file >> clusters_num;

    // initialize the cluster means
    NCC_clusters::cluster_means.reserve(clusters_num);

    // read the cluster means from the files
    for (int i = 0; i < clusters_num; i++) {
        // open the cluster mean file
        std::ifstream mean_file(cluster_dir + "/mean_cluster_" + std::to_string(i) + ".pgm");

        // read the pgm
        std::string magic_number;
        int width, height, max_value;
        mean_file >> magic_number >> width >> height >> max_value;

        // create the image
        auto *image = new MNIST_Image();

        // read the image data
        for (int j = 0; j < width * height; j++) {
            int pixel;
            mean_file >> pixel;
            image->setPixel(pixel, j);
        }

        // read the label from the stats file
        int label;
        stats_file >> label;

        // set the label
        image->setLabel(label);

        // add the mean to the cluster means
        NCC_clusters::cluster_means.push_back(image);
    }
}

/**
 * Destructor to free the memory allocated to the images
 */
NCC_clusters::~NCC_clusters() {
    // Delete the training images
    for (auto & training_image : NCC_clusters::training_images) {
        delete training_image;
    }

    // Delete the test images
    for (auto & test_image : NCC_clusters::test_images) {
        delete test_image;
    }

    if (NCC_clusters::from_file) {
        // Delete the cluster means
        for (auto & cluster_mean : NCC_clusters::cluster_means) {
            delete cluster_mean;
        }
    }

    delete generator;
}

// -------------- Setters -------------- //

/**
 * Increment the number of correct classifications
 */
void NCC_clusters::incrementCorrect() {
    NCC_clusters::n_correct++;
    NCC_clusters::n_tests++;
}

/**
 * Increment the number of incorrect classifications
 */
void NCC_clusters::incrementIncorrect() {
    NCC_clusters::n_incorrect++;
    NCC_clusters::n_tests++;
}

/**
 * Initialize the cluster centroids using k-means++
 */
void NCC_clusters::fitClusters(bool is_final, int dataset_fraction) {
    // create a random part of the training images
    std::vector<MNIST_Image *> random_training_images;
    random_training_images.reserve(NCC_clusters::training_images.size() / dataset_fraction);

    // assign random images from the training images to the random training images
    for (int i = 0; i < NCC_clusters::training_images.size() / dataset_fraction; i++) {
        random_training_images.push_back(NCC_clusters::training_images.at((*generator)() % NCC_clusters::training_images.size()));
    }

    // count the number of images in each cluster
    std::vector<int> cluster_counts(NCC_clusters::clusters.size(), 0);

    // Assign each training image to a cluster
    for (auto & training_image : random_training_images) {
        // find the distance to each cluster mean
        std::vector<double> distances;
        distances.reserve(NCC_clusters::n_clusters);

        for (auto & cluster_mean : NCC_clusters::cluster_means) {
            distances.push_back(training_image->calculateDistance(*cluster_mean));
        }

        // Find the minimum distance and assign the image reference to the corresponding cluster
        auto min_distance = std::min_element(distances.begin(), distances.end());
        int cluster_index = int(std::distance(distances.begin(), min_distance));

        if (is_final) {
            NCC_clusters::clusters.at(cluster_index).push_back(training_image);

            // increment the cluster count
            cluster_counts.at(cluster_index)++;

        } else {
            // increment the cluster count
            cluster_counts.at(cluster_index)++;
        }

        // Update the cluster based on the new image
        NCC_clusters::updateClusterMean(cluster_index, training_image, cluster_counts.at(cluster_index) - 1);
    }
}

/**
 * Update the cluster mean
 * @param cluster_index The index of the cluster
 * @param image         The image to add to the cluster
 */
void NCC_clusters::updateClusterMean(int cluster_index, MNIST_Image *image, int n_images) {
    // Calculate the new mean
    if (n_images == 0) {
        for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
            int old_sum = NCC_clusters::cluster_means.at(cluster_index)->getPixel(i);
            int new_pixel = image->getPixel(i);

            NCC_clusters::cluster_means.at(cluster_index)->setPixel((old_sum + new_pixel) / 2, i);
        }
    } else {
        for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
            int old_sum = NCC_clusters::cluster_means.at(cluster_index)->getPixel(i) * n_images;
            int new_pixel = image->getPixel(i);

            NCC_clusters::cluster_means.at(cluster_index)->setPixel((old_sum + new_pixel) / (n_images + 1), i);
        }
    }
}

/**
 * Determine the cluster label by finding the most common label in each cluster
 */
void NCC_clusters::determineClusterLabel() {
    // Find the most common label in each cluster
    for (int i = 0; i < NCC_clusters::n_clusters; i++) {
        // Count the number of times each label appears in the cluster
        std::vector<uint8_t> label_counts(10, 0);

        for (auto & image : NCC_clusters::clusters.at(i)) {
            label_counts.at(image->getLabel())++;
        }

        // Find the most common label
        auto max_count = std::max_element(label_counts.begin(), label_counts.end());
        int label = int(std::distance(label_counts.begin(), max_count));

        // Set the mean cluster label
        NCC_clusters::cluster_means.at(i)->setLabel(label);
    }
}

/**
 * Classify the test images using the cluster means
 */
int NCC_clusters::classifyImage(int test_index, bool verbose) {
    // Find the distance to each cluster mean
    std::vector<double> distances;
    distances.reserve(this->n_clusters);

    for (auto & cluster_mean : NCC_clusters::cluster_means) {
        distances.push_back(NCC_clusters::test_images.at(test_index)->calculateDistance(*cluster_mean));
    }

    // Find the minimum distance and assign the image reference to the corresponding cluster
    auto min_distance = std::min_element(distances.begin(), distances.end());
    int cluster_index = int(std::distance(distances.begin(), min_distance));

    // Update the statistics
    if (NCC_clusters::test_images.at(test_index)->getLabel() == NCC_clusters::cluster_means.at(cluster_index)->getLabel()) {
        NCC_clusters::incrementCorrect();
    } else {
        NCC_clusters::incrementIncorrect();
    }

    if (verbose) {
        std::cout << "Test image " << test_index << " is a " << int(test_images.at(test_index)->getLabel()) << std::endl;
        std::cout << "Test image " << test_index << " classified as " << int(NCC_clusters::cluster_means.at(cluster_index)->getLabel()) << std::endl;
    }

    return NCC_clusters::cluster_means.at(cluster_index)->getLabel();
}

/**
 * Print the classification statistics
 */
void NCC_clusters::printStats() {
    calculateAccuracy();  // Calculate the accuracy

    // Print the results
    std::cout << std::endl;
    std::cout << "Classification Summary:" << std::endl << std::endl;

    std::cout << "    Number of tests: " << n_tests << std::endl;
    std::cout << "    Number of correct classifications: " << n_correct << std::endl;
    std::cout << "    Number of incorrect classifications: " << n_incorrect << std::endl;
    std::cout.precision(3);
    std::cout << "    Accuracy: " << std::fixed << accuracy << "%" << std::endl;
}

/**
 * Calculate the accuracy
 */
void NCC_clusters::calculateAccuracy() {
    NCC_clusters::accuracy = (double) NCC_clusters::n_correct / (double) NCC_clusters::n_tests * 100;
}

/**
 * Save the mean cluster image as a .pgm file
 */
void NCC_clusters::saveMeanClusters() {
    // Create a stats file to save the number of clusters
    std::ofstream stats_file;
    stats_file.open("pre_fit/stats");
    stats_file << NCC_clusters::n_clusters << std::endl;

    // Save the mean clusters
    for (int i = 0; i < NCC_clusters::n_clusters; i++) {
        std::string filename = "pre_fit/mean_cluster_" + std::to_string(i);
        NCC_clusters::cluster_means.at(i)->saveImage(filename);

        // Save the class label to the stats file
        stats_file << int(NCC_clusters::cluster_means.at(i)->getLabel()) << std::endl;
    }

}

typedef struct {
    std::vector<double> *distances;  // Distances to each cluster mean
    int cluster_index;  // Index of the cluster with the minimum distance
    int thread_id;      // Thread ID
    int start_index;    // Start index of the training images
    int end_index;      // End index of the training images

    std::vector<MNIST_Image *> *images;  // Clusters
    NCC_clusters *clusters;  // Pointer to the NCC_clusters object
} CentroidArgs;

/**
 * Thread function for calculating the distances to the cluster means
 */
void *centroidThread(void *arg){
    auto *args = (CentroidArgs *) arg;
    NCC_clusters *clusters = args->clusters;

    for (int i = args->start_index; i < args->end_index; i++) {
//        MNIST_Image *training_image = clusters->training_images.at(i);
        MNIST_Image *training_image = args->images->at(i);

        // Find the distance to the nearest centroid
        double min_distance = std::numeric_limits<double>::max();

        for (auto & cluster_mean : clusters->cluster_means) {
            double distance = training_image->calculateDistance(*cluster_mean);

            if (distance < min_distance) {
                min_distance = distance;
            }
        }

        // Keep the minimum distance for each training image
        args->distances->push_back(min_distance);
    }

    pthread_exit(nullptr);
}

/**
 * Initialise the cluster means using the k-means++ algorithm
 *
 * @param dataset_fraction Fraction of the training dataset to use for initialisation
 */
void NCC_clusters::initializeCentroids(int dataset_fraction) {
    // Select the first centroid at random
    int first_centroid = int((*generator)() % NCC_clusters::training_images.size());
    NCC_clusters::cluster_means.push_back(NCC_clusters::training_images.at(first_centroid));

    progressbar p_bar((NCC_clusters::n_clusters - 1) * 60);

    // Select the remaining centroids
    for (int i = 1; i < NCC_clusters::n_clusters; i++) {
        // create a random part of the training images
        auto *random_training_images = new std::vector<MNIST_Image *>;
        random_training_images->reserve(NCC_clusters::training_images.size() / dataset_fraction);

        // assign random images from the training images to the random training images
        for (int j = 0; j < NCC_clusters::training_images.size() / dataset_fraction; j++) {
            random_training_images->push_back(NCC_clusters::training_images.at((*generator)() % NCC_clusters::training_images.size()));
        }

        // Calculate the distance to the nearest centroid for each training image
        std::vector<std::vector<double>> thread_distances;
        thread_distances.reserve(N_THREADS);

        for (int j = 0; j < N_THREADS; ++j) {
            thread_distances.emplace_back();
        }

        // thread attributes
        pthread_t threads[N_THREADS];
        CentroidArgs args[N_THREADS];

        // create the threads
        for (int j = 0; j < N_THREADS; j++) {
            args[j].distances = &thread_distances.at(j);
            args[j].distances->reserve(random_training_images->size() / N_THREADS);
            args[j].cluster_index = i;
            args[j].thread_id = j;
            args[j].start_index = int(j * (random_training_images->size() / N_THREADS));
            args[j].end_index = int((j + 1) * (random_training_images->size() / N_THREADS));
            args[j].images = random_training_images;
            args[j].clusters = this;

            pthread_create(&threads[j], nullptr, centroidThread, (void *) &args[j]);
        }

        // join the threads
        for (unsigned long thread : threads) {
            pthread_join(thread, nullptr);
        }

        // accumulate the distances
        std::vector<double> distances;
        distances.reserve(random_training_images->size());

        for (int j = 0; j < N_THREADS; j++) {
            distances.insert(distances.end(), thread_distances.at(j).begin(), thread_distances.at(j).end());
        }

        // Find the maximum distance of all the training images from all the centroids
        auto max_distance = std::max_element(distances.begin(), distances.end());
        int max_distance_index = int(std::distance(distances.begin(), max_distance));

        // Add the training image with the maximum distance to the cluster means
        NCC_clusters::cluster_means.push_back(random_training_images->at(max_distance_index));

        // Update the progress bar
        for (int j = 0; j < 60; ++j) {
            p_bar.update();
        }

        delete random_training_images;
    }
}

/**
 * Print the count of labels in each cluster
 *
 * @param cluster_index Index of the cluster
 */
void NCC_clusters::printClusterCounts(int cluster_index) {
    // Count the number of times each label appears in the cluster
    std::vector<int> label_counts(10, 0);

    for (auto & image : NCC_clusters::clusters.at(cluster_index)) {
        label_counts.at(image->getLabel())++;
    }

    std::cout << "Cluster " << cluster_index << " counts: " << std::endl;
    // Print the label counts
    for (int i = 0; i < 10; i++) {
        std::cout << "    Label " << i << ": " << label_counts.at(i) << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::endl;
}
