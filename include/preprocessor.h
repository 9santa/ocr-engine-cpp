#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

class DigitPreprocessor {
public:
    DigitPreprocessor();
    cv::Mat preprocess(const cv::Mat& input);
    std::vector<cv::Mat> extractDigits(const cv::Mat& image);

private:
    cv::Mat applyGrayscale(const cv::Mat& input);
    cv::Mat applyThreshold(const cv::Mat& input);
    cv::Mat removeNoise(const cv::Mat& input);
    std::vector<cv::Rect> findDigitContours(const cv::Mat& binary);
};





#endif // PREPROCESSOR_H
