#pragma once
#ifndef NEURAL_NETWORK_CLASSIFIER_H
#define NEURAL_NETWORK_CLASSIFIER_H

#include "baselines/common/training_sample.h"
#include "baselines/neural_network/nn/opt_mlp.h"
#include <string>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers = {400, 128, 64, 10});

    void train(const std::vector<TrainingSample>& trainingData, int epochs = 20, float learningRate = 0.05f);
    int predict_digit(const std::vector<float>& features);
    float evaluate(std::vector<TrainingSample>& testData);

    bool save_model(const std::string& filename) const;
    bool load_model(const std::string& filename);

private:
    OptMLP network;
    std::vector<int> layerSizes;

    // Convert features to ValPtr
    std::vector<OptValPtr> featuresToOptValPtr(GraphArena& arena, const std::vector<float>& features);

    // Convert label to one-hot encoding
    std::vector<OptValPtr> one_hot(GraphArena& arena, int label, int num_classes);

};

#endif // !NEURAL_NETWORK_CLASSIFIER_H
