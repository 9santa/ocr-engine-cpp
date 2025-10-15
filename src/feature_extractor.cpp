#include "feature_extractor.h"
#include <numeric>

FeatureExtractor::FeatureExtractor() {}

std::vector<float> FeatureExtractor::extractFeatures(const cv::Mat& digit) {
    std::vector<float> features;

    // Combine multiple feature extraction methods
    auto pixelFeatures = extractPixelFeatures(digit);
    auto zoningFeatures = extractZoningFeatures(digit);
    auto projectionFeatures = extractProjectionFeatures(digit);

    features.insert(features.end(), pixelFeatures.begin(), pixelFeatures.end());
    features.insert(features.end(), zoningFeatures.begin(), zoningFeatures.end());
    features.insert(features.end(), projectionFeatures.begin(), projectionFeatures.end());

    return features;
}



