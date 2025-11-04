#pragma once
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "image_matrix.h"
#include <vector>

class FeatureExtractor {
public:
    FeatureExtractor();

    // main feature extractor method, combines all feature types
    std::vector<float> extractFeatures(const ImageMatrix& digit);

    // individual feature extraction methods
    std::vector<float> extractPixelFeatures(const ImageMatrix& digit);
    std::vector<float> extractZoningFeatures(const ImageMatrix& digit);
    std::vector<float> extractProjectionFeatures(const ImageMatrix& digit);

    // utility methods
    void normalizeFeatures(std::vector<float>& features);
    
    // returns total feature vector size
    int getFeatureDimensions() const;

private:
    int zoningGridSize = 4; // 4x4 grid for zoning features

};


#endif // !FEATURE_EXTRACTOR_H
