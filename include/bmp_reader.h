#pragma once
#ifndef BMP_READER_H
#define BMP_READER_H

#include "image_matrix.h"
#include <cstdint>
#include <string>

class BMPReader {
public:
    static bool loadBMP(const std::string& path, ImageMatrix& image);
    static bool saveBMP(const std::string& path, const ImageMatrix& image);

private:
    #pragma pack(push, 1)
    struct BMPFileHeader {
        uint16_t signature;
        uint32_t file_size;
        uint16_t reserver1;
        uint16_t reserver2;
        uint32_t data_offset;
    };

    struct BMPInfoHeader {
        uint32_t header_size;
        int32_t width;
        int32_t height;
        uint16_t planes;
        uint16_t bit_count;
        uint32_t compression;
        uint32_t image_size;
        int32_t x_pixels_per_meter;
        int32_t y_pixels_per_meter;
        uint32_t colors_used;
        uint32_t colors_important;
    };
    #pragma pack(pop)

    static bool readHeaders(std::ifstream& file, BMPFileHeader& file_header, BMPInfoHeader& info_header);

};


#endif // !BMP_READER_H
