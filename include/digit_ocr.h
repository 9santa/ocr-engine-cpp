#pragma once
#ifndef DIGIT_OCR_H
#define DIGIT_OCR_H

#include "feature_extractor.h"
#include "knn_classifier.h"
#include "image_matrix.h"
#include "preprocessor.h"
#include "mnist_loader.h"
#include <vector>
#include <string>

class DigitOCR {
public:
    DigitOCR();

    void trainModel(const std::string& trainingDataPath);
    std::string recognize(const ImageMatrix& image);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

    // Testing (evaluation) methods
    float evaluateOnTestData(const std::string& testDataPath);
    float evaluateOnTrainingData();
    void confusionMatrix(const std::vector<MNISTImage>& testData);

    // Check if model is trained
    bool isTrained() const { return !trainingSamples.empty(); }

private:
    Preprocessor preprocessor;
    FeatureExtractor featureExtractor;
    KNNClassifier classifier;
    std::vector<TrainingSample> trainingSamples;

    std::vector<TrainingSample> loadMNISTData(const std::string& imagePath, const std::string& labelPath);
};


#endif // !DIGIT_OCR_H
