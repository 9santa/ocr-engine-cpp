#include "preprocessor.h"
#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

DigitPreprocessor::DigitPreprocessor() {}

cv::Mat DigitPreprocessor::preprocess(const cv::Mat& input) {
    cv::Mat processed = applyGrayscale(input);
    processed = applyThreshold(processed);
    processed = removeNoise(processed);
    return processed;
}

cv::Mat DigitPreprocessor::applyGrayscale(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    return gray;
}

cv::Mat DigitPreprocessor::applyThreshold(const cv::Mat& input) {
    cv::Mat binary;
    cv::adaptiveThreshold(input, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV, 11, 2);

    return binary;
}

cv::Mat DigitPreprocessor::removeNoise(const cv::Mat& input) {
    cv::Mat cleaned = input.clone();

    // remove small noise with morphological opening
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);

    return cleaned;
}
