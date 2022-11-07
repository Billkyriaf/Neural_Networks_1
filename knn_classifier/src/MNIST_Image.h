#ifndef KNN_CLASSIFIER_MNIST_IMAGE_H
#define KNN_CLASSIFIER_MNIST_IMAGE_H

#include <cstdint>
#include <array>

#define MNIST_IMAGE_SIZE (28 * 28)

class MNIST_Image {
public:
    // Constructors
    MNIST_Image() = default;     // Default constructor
    explicit MNIST_Image(uint8_t label);  // Constructor with label
    explicit MNIST_Image(uint8_t label, std::array<uint8_t, MNIST_IMAGE_SIZE> pixels);  // Constructor with label and pixels

    // Copy constructors

    // Destructor
    ~MNIST_Image() = default;

    // Getters
    uint8_t getLabel() const;  // Get label
    std::array<uint8_t, MNIST_IMAGE_SIZE> getPixels() const;  // Get pixels

    // Setters
    void setLabel(uint8_t label);  // Set label
    void setPixels(std::array<uint8_t, MNIST_IMAGE_SIZE> pixels);  // Set pixels

    // Functions
    double calculateDistance(const MNIST_Image &image) const;  // Calculate distance between this image and another image
    bool isLabel(uint8_t l) const;  // Check if the label of this image is the same as the given label

private:
    uint8_t label {0};                                   /// The label of the image
    std::array<uint8_t, MNIST_IMAGE_SIZE> pixels {};     /// The pixels of the image flattened into a 1D array
};


#endif
