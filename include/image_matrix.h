#pragma once
#ifndef IMAGE_MATRIX_H
#define IMAGE_MATRIX_H

#include <vector>
#include <cstdint>
#include <string>

class ImageMatrix {
public:
    int width, height, channels;
    std::vector<unsigned char> data;

    // constructors
    ImageMatrix();
    ImageMatrix(int w, int h, int c, unsigned char value = 0);

    // access
    inline unsigned char& operator()(int y, int x, int c);
    inline const unsigned char& operator()(int y, int x, int c) const;

    // basic utilities
    bool empty() const;
    void fill(unsigned char value);
    void resize(int new_width, int new_height, int new_channels = -1);

    // I/O (later)
    bool load(const std::string& path);
    bool save(const std::string& path);

    // conversions
    ImageMatrix to_grayscale() const;
};


#endif // !IMAGE_MATRIX_H
