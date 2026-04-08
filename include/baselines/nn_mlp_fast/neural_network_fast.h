#pragma once

#include "baselines/common/training_sample.h"
#include "baselines/nn_mlp_fast/dense_layer.h"
#include "baselines/nn_mlp_fast/matrix.h"
#include <string>
#include <vector>

class NeuralNetworkFast {
public:
    NeuralNetworkFast(int inputDim = 784, int hidden1 = 128, int hidden2 = 64, int numClasses = 10);

    void train(
        const std::vector<TrainingSample>& trainingData,
        int epochs = 10,
        float learningRate = 0.01f,
        std::size_t batchSize = 64,
        const std::string& runName = "");

    int predict_digit(const std::vector<float>& features) const;
    float evaluate(const std::vector<TrainingSample>& testData) const;

    bool save_model(const std::string& filename) const;
    bool load_model(const std::string& filename);

private:
    DenseLayer layer1;
    DenseLayer layer2;
    DenseLayer layer3;

    static Matrix makeBatchX(
        const std::vector<TrainingSample>& data,
        std::size_t start,
        std::size_t batchSize);

    static std::vector<int> makeBatchY(
        const std::vector<TrainingSample>& data,
        std::size_t start,
        std::size_t batchSize);

    static Matrix tanhForward(const Matrix& x);
    static Matrix tanhBackward(const Matrix& activated, const Matrix& gradOutput);
};
