#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureExtractor {
public:
    FeatureExtractor();
    std::vector<float> extractFeatures(const cv::Mat& digit);

private:
    std::vector<float> extractPixelFeatures(const cv::Mat& digit);
    std::vector<float> extractZoningFeatures(const cv::Mat& digit);
    std::vector<float> extractProjectionFeatures(const cv::Mat& digit);

};


#endif // !FEATURE_EXTRACTOR_H
