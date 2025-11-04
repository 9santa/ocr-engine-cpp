#pragma once
#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "image_matrix.h"
#include <vector>

struct BoundingBox {
    int x, y, width, height;
};

class Preprocessor {
public:
    Preprocessor();

    // main preprocessing
    ImageMatrix preprocess(const ImageMatrix& input);

    // digit extraction from full image
    std::vector<ImageMatrix> extractDigits(const ImageMatrix& image);

    // individual processing steps
    ImageMatrix applyGrayscale(const ImageMatrix& input);
    ImageMatrix applyThreshold(const ImageMatrix& input);
    // ImageMatrix applyAdapriveThreshold(const ImageMatrix& input, int blockSize = 11, double constant = 2);
    ImageMatrix removeNoise(const ImageMatrix& input);

    // contour and bounding box detection
    std::vector<BoundingBox> findDigitContours(const ImageMatrix& binary);

    // utility methods
    ImageMatrix resizeDigit(const ImageMatrix& digit, int targetWidth = 20, int targetHeight = 20);
    ImageMatrix normalizeDigit(const ImageMatrix& digit);

private:
    // helper functions for contour detection
    void findConnectedComponents(const ImageMatrix& binary, std::vector<std::vector<std::pair<int, int>>>& components);
    void DFS(int x, int y, const ImageMatrix& binary, std::vector<std::vector<bool>>& visited, std::vector<std::vector<std::pair<int, int>>>& component);

    // morphological preprocessing operations
    ImageMatrix morphologicalOperation(const ImageMatrix& input, const std::vector<std::vector<int>>& kernel, bool isDilation);

    // kernel for morphological operations
    std::vector<std::vector<int>> kernel = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 1, 0}
    };

    // size thresholds for digit filtering
    int minDigitWidth = 10;
    int minDigitHeight = 20;
    int maxDigitWidth = 100;
    int maxDigitHeight = 100;
};


#endif // PREPROCESSOR_H
