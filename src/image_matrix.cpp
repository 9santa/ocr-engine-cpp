#include "../include/image_matrix.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>

// constructors
ImageMatrix::ImageMatrix() : width(0), height(0), channels(0) {}

ImageMatrix::ImageMatrix(int w, int h, int c, unsigned char value)
    : width(w), height(h), channels(c), data(w*h*c, value) {}

// access operators
inline unsigned char& ImageMatrix::operator()(int y, int x, int c) {
    return data[(y * width + x) * channels + c];
}

inline const unsigned char& ImageMatrix::operator()(int y, int x, int c) const {
    return data[(y * width + x) * channels + c];
}

bool ImageMatrix::empty() const {
    return data.empty();
}

void ImageMatrix::fill(unsigned char value) {
    std::fill(data.begin(), data.end(), value);
}

void ImageMatrix::resize(int new_width, int new_height, int new_channels) {
    // new resized data vector
    std::vector<unsigned char> new_data(new_width * new_height * new_channels, 0);

    int copy_width = std::min(new_width, width);
    int copy_height = std::min(new_height, height);
    int copy_channles = std::min(new_channels, channels);

    for (int y = 0; y < copy_height; y++) {
        for (int x = 0; x < copy_width; x++) {
            for (int c = 0; c < copy_channles; c++) {
                new_data[(y * new_width + x) * new_channels + c] = data[(y * width + x) * channels + c];
            }
        }
    }

    width = new_width;
    height = new_height;
    channels = new_channels;
    data = std::move(new_data);
}
