#include "feature_extractor.h"
#include <numeric>
#include <ranges>

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

std::vector<float> FeatureExtractor::extractPixelFeatures(const cv::Mat& digit) {
    std::vector<float> features;
    features.reserve(digit.rows + digit.cols);

    for (int i = 0; i < digit.rows; i++) {
        for (int j = 0; j < digit.cols; j++) {
            features.push_back(digit.at<uchar>(i, j) / 255.0f);
        }
    }

    return features;
}

std::vector<float> FeatureExtractor::extractZoningFeatures(const cv::Mat& digit) {
    const int zones = 4;    // 4x4 grid
    const int zoneHeight = digit.rows / zones;
    const int zoneWidth = digit.cols / zones;

    std::vector<float> features;
    features.reserve(zones * zones);

    for (int i = 0; i < zones; i++) {
        for (int j = 0; j < zones; j++) {
            int startY = i * zoneHeight;
            int startX = j * zoneWidth;
            int endY = std::min((j + 1) * zoneHeight, digit.rows);
            int endX = std::min((j + 1) * zoneWidth, digit.cols);

            float zoneSum = 0;
            int pixelCount = 0;

            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    zoneSum += digit.at<uchar>(y, x);
                    pixelCount++;
                }
            }

            features.push_back(zoneSum / (pixelCount * 255.0f));
        }
    }

    return features;
}

std::vector<float> FeatureExtractor::extractProjectionFeatures(const cv::Mat& digit) {
    std::vector<float> features;

    // Horizontal projection
    std::vector<float> horizontalProj(digit.rows, 0);
    for (int i = 0; i < digit.rows; i++) {
        for (int j = 0; j < digit.cols; j++) {
            horizontalProj[i] += digit.at<uchar>(i, j);
        }
        horizontalProj[i] /= (digit.cols * 255.0f);
    }

    // Vertical projection
    std::vector<float> verticalProj(digit.cols, 0);
    for (int j = 0; j < digit.cols; j++) {
        for (int i = 0; i < digit.rows; i++) {
            verticalProj[j] += digit.at<uchar>(i, j);
        }
        verticalProj[j] /= (digit.rows * 255.0f);
    }

    features.insert(features.end(), horizontalProj.begin(), horizontalProj.end());
    features.insert(features.end(), verticalProj.begin(), verticalProj.end());

    return features;
}


