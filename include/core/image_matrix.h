#pragma once
#ifndef IMAGE_MATRIX_H
#define IMAGE_MATRIX_H

#include <vector>
#include <string>

class ImageMatrix {
public:
    int width, height, channels;
    std::vector<unsigned char> data;

    // constructors
    ImageMatrix();
    ImageMatrix(int w, int h, int c, unsigned char value = 0);

    // access
    inline unsigned char& operator()(int y, int x, int c) {
        return data[(y * width + x) * channels + c];
    }

    inline const unsigned char& operator()(int y, int x, int c) const {
        return data[(y * width + x) * channels + c];
    }
    // basic utilities
    bool empty() const;
    void fill(unsigned char value);
    void resize(int new_width, int new_height, int new_channels = -1);

    // I/O (later)
    bool load(const std::string& path);
    bool save(const std::string& path);

    // conversions (absolete, implemented in Prepocessor class)
    ImageMatrix to_grayscale() const;
};


#endif // !IMAGE_MATRIX_H
