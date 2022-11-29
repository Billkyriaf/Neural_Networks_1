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
 * Constructor with label and pixels
 *
 * @param label   The label of the image
 * @param pixels  The pixels of the image flattened into a 1D array
 */
MNIST_Image::MNIST_Image(uint8_t label, const std::array<uint8_t, MNIST_IMAGE_SIZE> pixels) : label(label), pixels(pixels) {
    // Normalize the pixels
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
        MNIST_Image::normalized_pixels[i] = (double) pixels[i] / 255.0;
    }
}

/**
 * Copy constructor
 *
 * @param other  The image to copy
 */
MNIST_Image::MNIST_Image(const MNIST_Image &other) {
    MNIST_Image::label = other.label;
    MNIST_Image::pixels = other.pixels;

    // Normalize the pixels
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
        MNIST_Image::normalized_pixels[i] = (double) MNIST_Image::pixels[i] / 255.0;
    }
}


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
 * Get the pixel at the given index
 *
 * @param index  The index of the pixel
 * @return The pixel at the given index
 */
uint8_t MNIST_Image::getPixel(int index) const {
    return MNIST_Image::pixels.at(index);
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
 * Set a pixel of the image. This will also update the normalized pixel at the same index
 *
 * @param pixel   The pixel to set
 * @param index   The index of the pixel to set
 */
void MNIST_Image::setPixel(uint8_t pixel, uint32_t index) {
    MNIST_Image::pixels[index] = pixel;

    // Normalize the pixel
    MNIST_Image::normalized_pixels[index] = (double) pixel / 255.0;
}

/**
 * Set the pixels of the image. This will also update the normalized pixels
 *
 * @param pixels   The pixels of the image
 */
void MNIST_Image::setPixels(std::array<uint8_t, MNIST_IMAGE_SIZE> p) {
    MNIST_Image::pixels = p;

    // Normalize the pixels
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
        MNIST_Image::normalized_pixels[i] = (double) MNIST_Image::pixels[i] / 255.0;
    }
}


// ------------- Member functions ------------- //
/**
 * Save the image to a pgm file
 *
 * @param name   The name of the file to save the image to
 */
void MNIST_Image::saveImage(const std::string &name, int max) const{
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
    out << "P2\n28 28\n" << max << "\n";

    if (max == 255){
        // Write the pixels
        for (int x = 0; x < 28; ++x) {
            for (int y = 0; y < 28; ++y) {
                auto pixel_val = (unsigned char)pixels[x * 28 + y];
                out << (unsigned int)pixel_val;
                out << " ";
            }
            out << "\n";
        }
    } else {
        // Write the pixels
        for (int x = 0; x < 28; ++x) {
            for (int y = 0; y < 28; ++y) {
                auto pixel_val = (double)normalized_pixels[x * 28 + y];
                out << (double)pixel_val;
                out << " ";
            }
            out << "\n";
        }
    }


    // Close the file
    out.close();
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


double *MNIST_Image::getPixelPtr(int index) {
    return &(MNIST_Image::normalized_pixels.at(index));
}

double MNIST_Image::getNormalizedPixel(int index) const {
    return MNIST_Image::normalized_pixels.at(index);
}

std::array<double, MNIST_IMAGE_SIZE> MNIST_Image::getNormalizedPixels() const {
    return MNIST_Image::normalized_pixels;
}

void MNIST_Image::setNormalizedPixel(double pixel, uint32_t index) {
    MNIST_Image::normalized_pixels[index] = pixel;

    // Un-normalize the pixel
    MNIST_Image::pixels[index] = (uint8_t) round(pixel * 255.0);
}

void MNIST_Image::setNormalizedPixels(std::array<double, 28 * 28> set_pixels) {
    MNIST_Image::normalized_pixels = set_pixels;

    // Un-normalize the set_pixels
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
        MNIST_Image::pixels[i] = (uint8_t) round(set_pixels[i] * 255.0);
    }
}
