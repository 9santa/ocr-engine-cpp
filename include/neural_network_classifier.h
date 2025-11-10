#pragma once
#ifndef NEURAL_NETWORK_CLASSIFIER_H
#define NEURAL_NETWORK_CLASSIFIER_H

#include "neural_network.h"
#include "knn_classifier.h"
#include <memory>


class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers = {784, 128, 64, 10});
    void train(const std::vector<TrainingSample>& trainingData, int epochs = 50, float learningRate = 0.05);
    int predict_digit(const std::vector<float>& features);
    float evaluate(const std::vector<TrainingSample>& testData);

private:
    std::unique_ptr<MLP> network;
    std::vector<int> layerSizes;

    // Convert features to ValPtr
    std::vector<ValPtr> featuresToValPtr(const std::vector<float>& features);

    // Convert label to one-hot encoding
    std::vector<ValPtr> one_hot(int label, int num_classes);

    // Softmax function
    std::vector<float> softMax(const std::vector<ValPtr>& outputs);
};


#endif // !NEURAL_NETWORK_CLASSIFIER_H
