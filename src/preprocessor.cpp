#include "preprocessor.h"
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

DigitPreprocessor::DigitPreprocessor() {}

cv::Mat DigitPreprocessor::preprocess(const cv::Mat& input) {
    cv::Mat processed = applyGrayscale(input);
    processed = applyThreshold(processed);
    processed = removeNoise(processed);
    return processed;
}

// Grayslace to simplify processing, reduces data from 3 channels to 1
cv::Mat DigitPreprocessor::applyGrayscale(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3) {
        // Gray = 0.299*R + 0.587*G + o.114*B
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    return gray;
}

// Binary Conversion.
// Converts grayscale image to binary (black&white) using adaptive thresholding
cv::Mat DigitPreprocessor::applyThreshold(const cv::Mat& input) {
    cv::Mat binary;
    cv::adaptiveThreshold(input, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV, 11, 2);

    return binary;
}

// Removes small noise particles using morphological operations
cv::Mat DigitPreprocessor::removeNoise(const cv::Mat& input) {
    cv::Mat cleaned = input.clone();

    // remove small noise with morphological opening
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);

    return cleaned;
}

// Digit Detection
std::vector<cv::Rect> DigitPreprocessor::findDigitContours(const cv::Mat& binary) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> digitRects;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);

        // filter contours by size to remove noise
        if (rect.width > 10 && rect.height > 20 && rect.width < binary.cols / 2 
                && rect.height < binary.rows / 2) {
            digitRects.push_back(rect);
        }
    }

    // sort rectangles from left to right
    std::sort(digitRects.begin(), digitRects.end(),
            [](const cv::Rect& a, const cv::Rect& b) {
                return a.x < b.x;
            });

    return digitRects;
}

std::vector<cv::Mat> DigitPreprocessor::extractDigits(const cv::Mat& image) {
    cv::Mat processed = preprocess(image);
    std::vector<cv::Rect> digitRects = findDigitContours(processed);

    std::vector<cv::Mat> digits;
    for (const auto& rect : digitRects) {
        cv::Mat digit = processed(rect).clone();

        // Resize to standard size for feature extraction
        cv::Mat resized;
        cv::resize(digit, resized, cv::Size(20, 20));

        digits.push_back(resized);
    }

    return digits;
}
