#include "MNIST_Image.h"
#include <cmath>

// ------------- Constructors ------------- //
/**
 * Constructor with label
 *
 * @param label   The label of the image
 */
MNIST_Image::MNIST_Image(uint8_t label) : label(label) {}

/**
 * Constructor with label and pixels
 *
 * @param label   The label of the image
 * @param pixels  The pixels of the image flattened into a 1D array
 */
MNIST_Image::MNIST_Image(uint8_t label, const std::array<uint8_t, 28 * 28> pixels) : label(label), pixels(pixels) {}


// ------------- Getters ------------- //
/**
 * Get the label of the image
 *
 * @return The label of the image
 */
uint8_t MNIST_Image::getLabel() const {
    return MNIST_Image::label;
}

/**
 * Get the distance between this image and another image
 *
 * @return The distance between this image and another image
 */
double MNIST_Image::getDistance() const {
    return MNIST_Image::distance;
}

/**
 * Get the pixels of the image
 *
 * @return The pixels of the image
 */
std::array<uint8_t, MNIST_IMAGE_SIZE> MNIST_Image::getPixels() const {
    return MNIST_Image::pixels;
}


// ------------- Setters ------------- //
/**
 * Set the label of the image
 *
 * @param label   The label of the image
 */
void MNIST_Image::setLabel(uint8_t l) {
    MNIST_Image::label = l;
}

/**
 * Set the distance between this image and the test image
 *
 * @param d  The distance between this image and the test image
 */
void MNIST_Image::setDistance(double d) {
    MNIST_Image::distance = d;
}

/**
 * Set a pixel of the image
 *
 * @param pixel   The pixel to set
 * @param index   The index of the pixel to set
 */
void MNIST_Image::setPixel(uint8_t pixel, uint32_t index) {
    MNIST_Image::pixels[index] = pixel;
}

/**
 * Set the pixels of the image
 *
 * @param pixels   The pixels of the image
 */
void MNIST_Image::setPixels(std::array<uint8_t, MNIST_IMAGE_SIZE> p) {
    MNIST_Image::pixels = p;
}


// ------------- Member functions ------------- //
/**
 * Calculate the distance between this image and another image. The distance is calculated using the Euclidean distance.
 * For performance reasons, the final square root is not calculated. The square root is a strictly increasing function.
 * If d1 > d2, then sqrt(d1) > sqrt(d2),so since we only want to compare the distances, we can safely ignore the square
 * root.
 *
 * @param image   The image to calculate the distance to
 * @return        The distance between this image and the other image
 */
double MNIST_Image::calculateDistance(const MNIST_Image &train_image){
    MNIST_Image::distance = 0;

    // Sum the squared differences between the pixels
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
        MNIST_Image::distance += std::pow(MNIST_Image::pixels[i] - train_image.pixels[i], 2);
    }

    return MNIST_Image::distance;
}


/**
 * Check if the label of this image is the same as the given label
 *
 * @param l   The label to check
 * @return    True if the label of this image is the same as the given label, false otherwise
 */
bool MNIST_Image::isLabel(uint8_t l) const {
    return MNIST_Image::label == l;
}
