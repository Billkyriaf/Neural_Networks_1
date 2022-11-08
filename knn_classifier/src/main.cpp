#include <iostream>
#include <cmath>

#include "MNIST_Import.h"


void save_image(const ::std::string &name, std::array<uint8_t, 28 * 28> image){
    using ::std::string;
    using ::std::ios;
    using ::std::ofstream;

    auto as_pgm = [](const string &name) -> string {
        if (! ((name.length() >= 4)
               && (name.substr(name.length() - 4, 4) == ".pgm")))
        {
            return name + ".pgm";
        } else {
            return name;
        }
    };

    ofstream out(as_pgm(name), ios::binary | ios::out | ios::trunc);

    out << "P2\n28 28\n255\n";
    for (int x = 0; x < 28; ++x) {
        for (int y = 0; y < 28; ++y) {
            auto pixel_val = (unsigned char)image[x * 28 + y];
            out << (unsigned int)pixel_val;
            out << " ";
        }
        out << "\n";
    }

    out.close();
}


int main() {
    MNIST_Import mnist(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte",
            "data/t10k-images.idx3-ubyte",
            "data/t10k-labels.idx1-ubyte"
    );

    mnist.readMetadata();  // Read the metadata from the file
    mnist.printMetadata();

    // The vector to store the training images
    std::vector<MNIST_Image *> training_images;
    training_images.reserve(mnist.getTrDataCount());

    mnist.readTrainingData(training_images);  // Read the training data

    // The vector to store the test images
    std::vector<MNIST_Image *> test_images;
    test_images.reserve(mnist.getTsDataCount());

    mnist.readTestData(test_images);  // Read the test data

    // Save the first image from the training set and the first image from the test set as pgm files to confirm that
    // the data was read correctly
    std::cout << "Saving training image as 'train_0.pgm'... Label: " << (int)training_images.at(0)->getLabel() << std::endl;
    save_image("images/train_0", training_images.at(0)->getPixels());

    std::cout << "Saving test image as 'test_1.pgm'... Label: " << (int)test_images.at(0)->getLabel() << std::endl;
    save_image("images/test_0", test_images.at(0)->getPixels());

    // At this point the data is stored in the vectors and the KNN algorithm can start


    // Free the memory
    for (int i = 0; i < mnist.getTrDataCount(); ++i) {
        delete training_images.at(i);
    }

    for (int i = 0; i < mnist.getTsDataCount(); ++i) {
        delete test_images.at(i);
    }

    return 0;
}
