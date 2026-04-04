#include "baselines/knn/feature_extractor.h"

FeatureExtractor::FeatureExtractor() {}

std::vector<float> FeatureExtractor::extractKNNFeatures(const ImageMatrix& digit) const {
    std::vector<float> features;

    // Combine multiple feature types
    auto pixelFeatures = extractPixelFeatures(digit);
    auto zoningFeatures = extractZoningFeatures(digit);
    auto projectionFeatures = extractProjectionFeatures(digit);

    features.reserve(pixelFeatures.size() + zoningFeatures.size() + projectionFeatures.size());
    features.insert(features.end(), pixelFeatures.begin(), pixelFeatures.end());
    features.insert(features.end(), zoningFeatures.begin(), zoningFeatures.end());
    features.insert(features.end(), projectionFeatures.begin(), projectionFeatures.end());

    return features;
}

std::vector<float> FeatureExtractor::extractPixelFeatures(const ImageMatrix& digit) const {
    std::vector<float> features;
    features.reserve(digit.width * digit.height);

    for (int y = 0; y < digit.height; y++) {
        for (int x = 0; x < digit.width; x++) {
            // normalize pixel values to [0, 1]
            features.push_back(digit(y, x, 0) / 255.0f);
        }
    }

    return features;
}

std::vector<float> FeatureExtractor::extractZoningFeatures(const ImageMatrix& digit) const {
    const int zones = zoningGridSize;    // default=4x4 grid
    const int zoneHeight = digit.height / zones;
    const int zoneWidth = digit.width / zones;

    std::vector<float> features;
    features.reserve(zones * zones);

    for (int i = 0; i < zones; i++) {
        for (int j = 0; j < zones; j++) {
            int startY = i * zoneHeight;
            int startX = j * zoneWidth;
            int endY = std::min((i + 1) * zoneHeight, digit.height);
            int endX = std::min((j + 1) * zoneWidth, digit.width);

            float zoneSum = 0.0f;
            int pixelCount = 0;

            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    zoneSum += digit(y, x, 0) / 255.0f;
                    pixelCount++;
                }
            }

            features.push_back(pixelCount > 0 ? zoneSum / pixelCount : 0.0f);
        }
    }

    return features;
}

std::vector<float> FeatureExtractor::extractProjectionFeatures(const ImageMatrix& digit) const {
    std::vector<float> features;
    features.reserve(digit.height + digit.width);

    // Horizontal projection
    std::vector<float> horizontalProj(digit.height, 0.0f);
    for (int y = 0; y < digit.height; y++) {
        for (int x = 0; x < digit.width; x++) {
            horizontalProj[y] += digit(y, x, 0) / 255.0f;
        }
        horizontalProj[y] /= digit.width;
    }

    // Vertical projection
    std::vector<float> verticalProj(digit.width, 0.0f);
    for (int x = 0; x < digit.width; x++) {
        for (int y = 0; y < digit.height; y++) {
            verticalProj[x] += digit(y, x, 0) / 255.0f;
        }
        verticalProj[x] /= digit.height;
    }

    features.insert(features.end(), horizontalProj.begin(), horizontalProj.end());
    features.insert(features.end(), verticalProj.begin(), verticalProj.end());

    return features;
}

int FeatureExtractor::getKNNFeatureDimensions(int width, int height) const {
    return width * height + zoningGridSize * zoningGridSize + width + height;
}

std::vector<float> FeatureExtractor::extractNeuralNetworkFeatures(const ImageMatrix& digit) const {
    return extractPixelFeatures(digit);
}
