#pragma once
#ifndef DIGIT_OCR_H
#define DIGIT_OCR_H

#include "neural_network_classifier.h"
#include "feature_extractor.h"
#include "knn_classifier.h"
#include "image_matrix.h"
#include "preprocessor.h"
#include "mnist_loader.h"
#include <vector>
#include <string>

enum AlgorithmType {
    KNN,
    NEURAL_NETWORK
};

class DigitOCR {
public:
    DigitOCR();

    void trainModel(const std::string& trainingDataPath, AlgorithmType algo = AlgorithmType::KNN);
    std::string recognize(const ImageMatrix& image, AlgorithmType algo = AlgorithmType::KNN);
    void saveModel(const std::string& filename, AlgorithmType algo = AlgorithmType::KNN);
    void loadModel(const std::string& filename, AlgorithmType algo = AlgorithmType::KNN);

    // Testing (evaluation) methods
    float evaluateOnTestData(const std::string& testDataPath, AlgorithmType algo = AlgorithmType::KNN);
    float evaluateOnTrainingData();
    void confusionMatrix(const std::vector<MNISTImage>& testData, AlgorithmType algo = AlgorithmType::KNN);

    // Check if model is trained
    bool isTrained(AlgorithmType algo = AlgorithmType::KNN) const { return !trainingSamples.empty(); }

private:
    Preprocessor preprocessor;
    FeatureExtractor featureExtractor;
    KNNClassifier classifier;
    NeuralNetwork nnClassifier;
    std::vector<TrainingSample> trainingSamples;

    std::vector<TrainingSample> loadMNISTData(const std::string& imagePath, const std::string& labelPath);
};


#endif // !DIGIT_OCR_H
