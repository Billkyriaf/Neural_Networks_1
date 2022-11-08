#ifndef KNN_CLASSIFIER_MNIST_IMPORT_H
#define KNN_CLASSIFIER_MNIST_IMPORT_H

#include <vector>
#include <string>
#include <fstream>
#include "MNIST_Image.h"

class MNIST_Import {
public:
    // Constructors
    MNIST_Import() = default;  // Default constructor

    // Constructor with path to MNIST dataset files
    explicit MNIST_Import(std::string tr_data_p,
                          std::string tr_label_p,
                          std::string test_data_p,
                          std::string test_label_p);

    // Destructor
    ~MNIST_Import();

    // Getters

    // Setters

    // Functions
    void readMetadata();  // Read the metadata of the MNIST dataset

    void readTrainingData(std::vector<MNIST_Image *>& training_images);
    void readTestData(std::vector<MNIST_Image *>& test_images);


private:
    void openFile(const std::string& path, std::ifstream& stream);  // Open a file and return a file stream

    // Training data
    std::string tr_data_path {};         // Path to the training data file
    std::string tr_label_path {};        // Path to the training labels file

    std::ifstream tr_data_file;          // Training data file
    std::ifstream tr_label_file;         // Training labels file

    uint32_t tr_data_magic_number {0};   // Magic number of the training data file
    uint32_t tr_label_magic_number {0};  // Magic number of the training labels file
    uint32_t tr_data_count {0};          // Number of images in the training data file
    uint32_t tr_label_count {0};         // Number of labels in the training labels file
    uint32_t tr_data_rows {0};           // Number of rows in the training data file
    uint32_t tr_data_cols {0};           // Number of columns in the training data file


    // Test data
    std::string ts_data_path {};         // Path to the test data file
    std::string ts_label_path {};        // Path to the test labels file

    std::ifstream ts_data_file;          // Test data file
    std::ifstream ts_label_file;         // Test labels file

    uint32_t ts_data_magic_number {0};   // Magic number of the test data file
    uint32_t ts_label_magic_number {0};  // Magic number of the test labels file
    uint32_t ts_data_count {0};          // Number of images in the test data file
    uint32_t ts_label_count {0};         // Number of labels in the test labels file
    uint32_t ts_data_rows {0};           // Number of rows in the test data file
    uint32_t ts_data_cols {0};           // Number of columns in the test data file

};


#endif
