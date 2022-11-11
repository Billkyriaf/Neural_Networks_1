#ifndef KNN_CLASSIFIER_MNIST_IMAGE_H
#define KNN_CLASSIFIER_MNIST_IMAGE_H

#include <cstdint>
#include <array>
#include <iostream>

#define MNIST_IMAGE_SIZE (28 * 28)

class MNIST_Image {
public:
    // Constructors
    MNIST_Image() = default;
    explicit MNIST_Image(uint8_t label);
    explicit MNIST_Image(uint8_t label, std::array<uint8_t, MNIST_IMAGE_SIZE> pixels);

    // Copy constructors
    MNIST_Image(const MNIST_Image &other);

    // Destructor
    ~MNIST_Image() = default;

    // Getters
    uint8_t getLabel() const;
    double getDistance() const;
    uint8_t getPixel(int index) const;
    std::array<uint8_t, MNIST_IMAGE_SIZE> getPixels() const;

    // Setters
    void setLabel(uint8_t label);
    void setDistance(double d);
    void setPixel(uint8_t pixel, uint32_t index);
    void setPixels(std::array<uint8_t, MNIST_IMAGE_SIZE> pixels);

    // Functions
    void saveImage(const ::std::string &name) const;
    bool isLabel(uint8_t l) const;
    double calculateDistance(const MNIST_Image &test_image);


private:
    // Variables
    uint8_t label {0};                                   /// The label of the image
    double distance {0};                                 /// The distance between this image and another image

    std::array<uint8_t, MNIST_IMAGE_SIZE> pixels{};      /// The pixels of the image flattened into a 1D array
};


#endif
