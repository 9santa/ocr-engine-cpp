#include "../include/bmp_reader.h"
#include <fstream>
#include <iostream>

bool BMPReader::readHeaders(std::ifstream& file, BMPFileHeader& file_header, BMPInfoHeader& info_header) {
    file.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));
    file.read(reinterpret_cast<char*>(&info_header), sizeof(info_header));

    // validate BMP signature
    if (file_header.signature != 0x4D42) { // 'BM' in little-endian
        std::cerr << "Invalid BMP file: missing 'BM' signature\n";
        return false;
    }

    // only support uncompressed 24-bit for now
    if (info_header.bit_count != 24 || info_header.compression != 0) {
        std::cerr << "Unsupported BMP format (only 24-bit RGB uncompressed supported)\n";
        return false;
    }

    return true;
}

bool BMPReader::loadBMP(const std::string &path, ImageMatrix &image) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open BMP file: " << path << "\n";
        return false;
    }

    BMPFileHeader file_header;
    BMPInfoHeader info_header;
    if (!readHeaders(file, file_header, info_header)) return false;

    // ImageMatrix to hold pixel data
    image.width = info_header.width;
    image.height = std::abs(info_header.height); // can be negative if upside-down
    image.channels = 3; // RGB
    image.data.resize(image.width * image.height * image.channels);

    // move to pixel data start
    file.seekg(file_header.data_offset, std::ios::beg);

    // each row is padded to 4-byte boundaries, round down to power of 2
    int row_padded = (image.width * 3 + 3) & (~3);
    std::vector<unsigned char> row_data(row_padded);

    bool bottomUp = (info_header.height > 0);

    for (int y = 0; y < image.height; y++) {
        int rowIndex = bottomUp ? (image.height - 1 - y) : y;
        file.read(reinterpret_cast<char*>(row_data.data()), row_padded);

        for (int x = 0; x < image.width; x++) {
            unsigned char b = row_data[x * 3 + 0];
            unsigned char g = row_data[x * 3 + 1];
            unsigned char r = row_data[x * 3 + 2];

            image(rowIndex, x, 0) = r;
            image(rowIndex, x, 1) = g;
            image(rowIndex, x, 2) = b;
        }
    }

    return true;
}

bool BMPReader::saveBMP(const std::string &path, const ImageMatrix &image) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot save BMP to " << path << "\n";
        return false;
    }

    int row_padded = (image.width * 3 + 3) & (~3);
    int image_size = row_padded * image.height;

    BMPFileHeader file_header = {
        0x4D42,
        static_cast<uint32_t>(sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size),
        0, 0,
        static_cast<uint32_t>(sizeof(BMPFileHeader) + sizeof(BMPInfoHeader))
    };

    BMPInfoHeader info_header = {
        sizeof(BMPInfoHeader),
        image.width,
        image.height,
        1,
        24,
        0,
        static_cast<uint32_t>(image_size),
        2835, 2835,
        0, 0
    };

    file.write(reinterpret_cast<char*>(&file_header), sizeof(file_header));
    file.write(reinterpret_cast<char*>(&info_header), sizeof(info_header));

    std::vector<unsigned char> row_data(row_padded, 0);
    for (int y = image.height-1; y >= 0; y--) {
        for (int x = 0; x < image.width; x++) {
            row_data[x*3+0] = image(y, x, 2);    // B
            row_data[x*3+1] = image(y, x, 1);    // G
            row_data[x*3+2] = image(y, x, 0);    // R
        }
        file.write(reinterpret_cast<char*>(row_data.data()), row_padded);
    }

    return true;
}
