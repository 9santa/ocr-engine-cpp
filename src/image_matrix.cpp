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
