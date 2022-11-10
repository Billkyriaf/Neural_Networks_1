#include "MNIST_Image.h"
#include <cmath>
#include <fstream>

// ------------- Constructors ------------- //
/**
 * Constructor with label
 *
 * @param label   The label of the image
 */
MNIST_Image::MNIST_Image(uint8_t label) : label(label) {}

/**
 * Copy constructor
 *
 * @param other  The image to copy
 */
MNIST_Image::MNIST_Image(const MNIST_Image &other) {
    label = other.label;
    distance = other.distance;
    pixels = other.pixels;
}

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
double MNIST_Image::calculateDistance(const MNIST_Image &test_image){
    MNIST_Image::distance = 0;

    // Sum the squared differences between the pixels
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
        MNIST_Image::distance += std::pow(MNIST_Image::pixels[i] - test_image.pixels[i], 2);
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


/**
 * Save the image to a pgm file
 *
 * @param name   The name of the file to save the image to
 */
void MNIST_Image::saveImage(const std::string &name) const{
    using ::std::string;
    using ::std::ios;
    using ::std::ofstream;

    // lambda function to create the file name
    auto as_pgm = [](const string &name) -> string {
        if (! ((name.length() >= 4)
               && (name.substr(name.length() - 4, 4) == ".pgm")))
        {
            return name + ".pgm";
        } else {
            return name;
        }
    };

    // Open the file for writing in binary mode
    ofstream out(as_pgm(name), ios::binary | ios::out | ios::trunc);

    // Write the header
    out << "P2\n28 28\n255\n";

    // Write the pixels
    for (int x = 0; x < 28; ++x) {
        for (int y = 0; y < 28; ++y) {
            auto pixel_val = (unsigned char)pixels[x * 28 + y];
            out << (unsigned int)pixel_val;
            out << " ";
        }
        out << "\n";
    }

    // Close the file
    out.close();
}

