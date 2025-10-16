#include "digit_ocr.h"
#include <fstream>
#include <iostream>

DigitOCR::DigitOCR() : classifier(3) {}

std::vector<TrainingSample> DigitOCR::loadMNISTData(const std::string& imagePath, const std::string& lapelPath) {
    std::vector<TrainingSample> samples;

    return samples;
}

void DigitOCR::trainModel(const std::string& trainingDataPath) {

    std::cout << "Training model... \n";
}

std::string DigitOCR::recognize(const cv::Mat& image) {
    auto digits = preprocessor.extractDigits(image);
    std::string result;

    for (const auto& digit : digits) {
        auto features = FeatureExtractor.extractFeatures(digit);
        int prediction = classifier.predict(features);
        result += std::to_string(prediction);
    }

    return result;
}

void DigitOCR::saveModel(const std::string& filename) {
    std::cout << "Saving model to " << filename << "\n";
}

void DigitOCR::loadModel(const std::string& filename) {
    std::cout << "Loading model from " << filename << "\n";
}
