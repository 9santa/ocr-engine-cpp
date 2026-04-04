#pragma once
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "core/image_matrix.h"
#include <vector>

class FeatureExtractor {
public:
    FeatureExtractor();

    // main KNN feature extractor method, combines all feature types
    std::vector<float> extractKNNFeatures(const ImageMatrix& digit) const;
    std::vector<float> extractNeuralNetworkFeatures(const ImageMatrix& digit) const;
    // individual feature extraction methods
    std::vector<float> extractPixelFeatures(const ImageMatrix& digit) const;
    std::vector<float> extractZoningFeatures(const ImageMatrix& digit) const;
    std::vector<float> extractProjectionFeatures(const ImageMatrix& digit) const;

    // returns total feature vector size
    int getKNNFeatureDimensions(int width, int height) const;

private:
    int zoningGridSize = 4; // 4x4 grid for zoning features
};

#endif // !FEATURE_EXTRACTOR_H
