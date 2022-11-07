#include "MNIST_Import.h"

#include <utility>
#include <iostream>
#include <endian.h>

/**
 * Constructor with path to MNIST dataset files
 *
 * @param tr_data_p     Path to the training data file
 * @param tr_label_p    Path to the training labels file
 * @param test_data_p   Path to the test data file
 * @param test_label_p  Path to the test labels file
 */
MNIST_Import::MNIST_Import(std::string tr_data_p, std::string tr_label_p, std::string test_data_p,
                           std::string test_label_p) : tr_data_path(std::move(tr_data_p)),
                                                       tr_label_path(std::move(tr_label_p)),
                                                       ts_data_path(std::move(test_data_p)),
                                                       ts_label_path(std::move(test_label_p)) {

}

/**
 * Destructor for MNIST_Import. Closes the file streams if they are open
 */
MNIST_Import::~MNIST_Import() {
    if (tr_data_file.is_open()) {
        tr_data_file.close();
    }

    if (tr_label_file.is_open()) {
        tr_label_file.close();
    }

    if (ts_data_file.is_open()) {
        ts_data_file.close();
    }

    if (ts_label_file.is_open()) {
        ts_label_file.close();
    }
}


/**
 * Open a binary file for reading
 *
 * @param path    Path to the file
 * @param stream  The file stream to open
 */
void MNIST_Import::openFile(const std::string &path, std::ifstream &stream) {
    // Open the file
    stream.open(path, std::ios::binary);

    // Check if the file was opened successfully
    if (!stream.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
}

void MNIST_Import::readMetadata() {

    openFile(tr_data_path, tr_data_file);  // Open the training data file

    // Read the magic number
    tr_data_file.read(reinterpret_cast<char *>(&tr_data_magic_number), sizeof(tr_data_magic_number));

    // Convert the magic number to host byte order
    tr_data_magic_number = be32toh(tr_data_magic_number);

    // Check if the magic number is correct
    if (tr_data_magic_number != 2051) {
        throw std::runtime_error("Invalid magic number in training data file got: " + std::to_string(tr_data_magic_number));
    }

    tr_data_file.read(reinterpret_cast<char *>(&tr_data_count), sizeof(tr_data_count));  // Read the number of images
    tr_data_count = be32toh(tr_data_count);  // Convert the number of images to host byte order

    tr_data_file.read(reinterpret_cast<char *>(&tr_data_rows), sizeof(tr_data_rows));  // Read the number of rows
    tr_data_rows = be32toh(tr_data_rows);  // Convert the number of rows to host byte order

    tr_data_file.read(reinterpret_cast<char *>(&tr_data_cols), sizeof(tr_data_cols));  // Read the number of columns
    tr_data_cols = be32toh(tr_data_cols);  // Convert the number of columns to host byte order

    openFile(tr_label_path, tr_label_file);  // Open the training labels file

    // Read the magic number
    tr_label_file.read(reinterpret_cast<char *>(&tr_label_magic_number), sizeof(tr_label_magic_number));

    // Convert the magic number to host byte order
    tr_label_magic_number = be32toh(tr_label_magic_number);

    // Check if the magic number is correct
    if (tr_label_magic_number != 2049) {
        throw std::runtime_error("Invalid magic number in training labels file");
    }

    tr_label_file.read(reinterpret_cast<char *>(&tr_label_count), sizeof(tr_label_count));  // Read the number of labels
    tr_label_count = be32toh(tr_label_count);  // Convert the number of labels to host byte order

    // Check if the number of images and labels match
    if (tr_data_count != tr_label_count) {
        throw std::runtime_error("Number of images and labels do not match");
    }


    openFile(ts_data_path, ts_data_file);  // Open the test data file

    // Read the magic number
    ts_data_file.read(reinterpret_cast<char *>(&ts_data_magic_number), sizeof(ts_data_magic_number));

    // Convert the magic number to host byte order
    ts_data_magic_number = be32toh(ts_data_magic_number);

    // Check if the magic number is correct
    if (ts_data_magic_number != 2051) {
        throw std::runtime_error("Invalid magic number in test data file");
    }

    ts_data_file.read(reinterpret_cast<char *>(&ts_data_count), sizeof(ts_data_count));  // Read the number of images
    ts_data_count = be32toh(ts_data_count);  // Convert the number of images to host byte order

    ts_data_file.read(reinterpret_cast<char *>(&ts_data_rows), sizeof(ts_data_rows));  // Read the number of rows
    ts_data_rows = be32toh(ts_data_rows);  // Convert the number of rows to host byte order

    ts_data_file.read(reinterpret_cast<char *>(&ts_data_cols), sizeof(ts_data_cols));  // Read the number of columns
    ts_data_cols = be32toh(ts_data_cols);  // Convert the number of columns to host byte order

    // Open the test labels file
    openFile(ts_label_path, ts_label_file);

    // Read the magic number
    ts_label_file.read(reinterpret_cast<char *>(&ts_label_magic_number), sizeof(ts_label_magic_number));

    // Convert the magic number to host byte order
    ts_label_magic_number = be32toh(ts_label_magic_number);

    // Check if the magic number is correct
    if (ts_label_magic_number != 2049) {
        throw std::runtime_error("Invalid magic number in test labels file");
    }

    ts_label_file.read(reinterpret_cast<char *>(&ts_label_count), sizeof(ts_label_count));  // Read the number of labels
    ts_label_count = be32toh(ts_label_count);  // Convert the number of labels to host byte order

    // Check if the number of images and labels match
    if (tr_data_count != tr_label_count) {
        throw std::runtime_error("Number of training images and labels do not match");
    }
}

void MNIST_Import::printMetadata() {
    std::cout << "Training data:" << std::endl;
    std::cout << "Magic number: " << tr_data_magic_number << std::endl;
    std::cout << "Number of images: " << tr_data_count << std::endl;
    std::cout << "Number of rows: " << tr_data_rows << std::endl;
    std::cout << "Number of columns: " << tr_data_cols << std::endl;
    std::cout << std::endl;

    std::cout << "Training labels:" << std::endl;
    std::cout << "Magic number: " << tr_label_magic_number << std::endl;
    std::cout << "Number of labels: " << tr_label_count << std::endl;
    std::cout << std::endl;

    std::cout << "Test data:" << std::endl;
    std::cout << "Magic number: " << ts_data_magic_number << std::endl;
    std::cout << "Number of images: " << ts_data_count << std::endl;
    std::cout << "Number of rows: " << ts_data_rows << std::endl;
    std::cout << "Number of columns: " << ts_data_cols << std::endl;
    std::cout << std::endl;

    std::cout << "Test labels:" << std::endl;
    std::cout << "Magic number: " << ts_label_magic_number << std::endl;
    std::cout << "Number of labels: " << ts_label_count << std::endl;
    std::cout << std::endl;
}

