#pragma once
#ifndef DIGIT_OCR_H
#define DIGIT_OCR_H

#include <string>
#include <vector>

#include "baselines/common/training_sample.h"
#include "baselines/knn/feature_extractor.h"
#include "baselines/knn/knn_classifier.h"
#include "baselines/neural_network/neural_network_classifier.h"
#include "core/image_matrix.h"
#include "data/mnist_loader.h"
#include "preprocess/preprocessor.h"

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
    bool isTrained(AlgorithmType algo = AlgorithmType::KNN) const;

private:
    Preprocessor preprocessor;
    FeatureExtractor featureExtractor;
    KNNClassifier classifier;
    NeuralNetwork nnClassifier;

    bool knnTrained = false;
    bool neuralNetworkTrained = false;

    std::vector<TrainingSample> knnTrainingSamples;

    std::vector<TrainingSample> loadMNISTSamples(const std::vector<MNISTImage>& data, AlgorithmType algo) const;
};


#endif // !DIGIT_OCR_H
