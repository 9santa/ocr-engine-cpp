#pragma once
#ifndef NEURAL_NETWORK_CLASSIFIER_H
#define NEURAL_NETWORK_CLASSIFIER_H

#include "opt_mlp.h"
#include "loss.h"
#include "knn_classifier.h"
#include <memory>


class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers = {784, 128, 64, 10});

    void train(const std::vector<TrainingSample>& trainingData, int epochs = 50, float learningRate = 0.05);
    int predict_digit(const std::vector<float>& features);
    float evaluate(std::vector<TrainingSample>& testData);

private:
    OptMLP network;
    std::vector<int> layerSizes;

    // Convert features to ValPtr
    std::vector<OptValPtr> featuresToOptValPtr(const std::vector<float>& features);

    // Convert label to one-hot encoding
    std::vector<OptValPtr> one_hot(int label, int num_classes);

};


#endif // !NEURAL_NETWORK_CLASSIFIER_H
