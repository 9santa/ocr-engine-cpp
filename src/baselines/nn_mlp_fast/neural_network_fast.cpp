#include "baselines/nn_mlp_fast/neural_network_fast.h"
#include "baselines/nn_mlp_fast/loss.h"
#include "experiments/training_logger.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>

NeuralNetworkFast::NeuralNetworkFast(int inputDim, int hidden1, int hidden2, int numClasses)
    : layer1(inputDim, hidden1),
      layer2(hidden1, hidden2),
      layer3(hidden2, numClasses) {}

Matrix NeuralNetworkFast::makeBatchX(
    const std::vector<TrainingSample>& data,
    std::size_t start,
    std::size_t batchSize) {

    const std::size_t end = std::min(start + batchSize, data.size());
    const std::size_t actualBatch = end - start;
    const int featureDim = static_cast<int>(data[start].features.size());

    Matrix X(static_cast<int>(actualBatch), featureDim, 0.0f);

    for (std::size_t r = 0; r < actualBatch; r++) {
        for (int c = 0; c < featureDim; c++) {
            X(static_cast<int>(r), c) = data[start + r].features[c];
        }
    }

    return X;
}

std::vector<int> NeuralNetworkFast::makeBatchY(
    const std::vector<TrainingSample>& data,
    std::size_t start,
    std::size_t batchSize) {

    const std::size_t end = std::min(start + batchSize, data.size());
    std::vector<int> y;
    y.reserve(end - start);

    for (std::size_t i = start; i < end; i++) {
        y.push_back(data[i].label);
    }

    return y;
}

Matrix NeuralNetworkFast::tanhForward(const Matrix& x) {
    Matrix out(x.rows, x.cols, 0.0f);

    for (int r = 0; r < x.rows; r++) {
        for (int c = 0; c < x.cols; c++) {
            out(r, c) = std::tanh(x(r, c));
        }
    }

    return out;
}

Matrix NeuralNetworkFast::tanhBackward(const Matrix& activated, const Matrix& gradOutput) {
    Matrix grad(activated.rows, activated.cols, 0.0f);

    for (int r = 0; r < activated.rows; r++) {
        for (int c = 0; c < activated.cols; c++) {
            grad(r, c) = gradOutput(r, c) * (1.0f - activated(r, c) * activated(r, c));
        }
    }

    return grad;
}

void NeuralNetworkFast::train(
    const std::vector<TrainingSample>& trainingData,
    int epochs,
    float learningRate,
    std::size_t batchSize,
    const std::string& runName) {

    if (trainingData.empty()) {
        return;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epochLoss = 0.0f;
        std::size_t batches = 0;

        for (std::size_t start = 0; start < trainingData.size(); start += batchSize) {
            Matrix X = makeBatchX(trainingData, start, batchSize);
            std::vector<int> y = makeBatchY(trainingData, start, batchSize);

            Matrix Z1 = layer1.forward(X);
            Matrix A1 = tanhForward(Z1);

            Matrix Z2 = layer2.forward(A1);
            Matrix A2 = tanhForward(Z2);

            Matrix logits = layer3.forward(A2);

            auto lossResult = softmaxCrossEntropyForward(logits, y);
            epochLoss += lossResult.loss;
            batches++;

            Matrix dLogits = softmaxCrossEntropyBackward(lossResult.probs, y);
            Matrix dA2 = layer3.backward(dLogits);
            Matrix dZ2 = tanhBackward(A2, dA2);

            Matrix dA1 = layer2.backward(dZ2);
            Matrix dZ1 = tanhBackward(A1, dA1);

            layer1.backward(dZ1);

            layer1.step(learningRate);
            layer2.step(learningRate);
            layer3.step(learningRate);
        }

        const float avgLoss = epochLoss / static_cast<float>(batches);

        std::cout << "Epoch " << (epoch + 1)
                  << "/" << epochs
                  << " - Avg Loss: " << avgLoss
                  << "\n";

        if (!runName.empty()) {
            TrainingLogger::logEpochLoss(runName, epoch + 1, avgLoss);
        }
    }
}

int NeuralNetworkFast::predict_digit(const std::vector<float>& features) const {
    Matrix X(1, static_cast<int>(features.size()), 0.0f);
    for (int i = 0; i < static_cast<int>(features.size()); i++) {
        X(0, i) = features[i];
    }

    auto layer1Copy = const_cast<DenseLayer&>(layer1);
    auto layer2Copy = const_cast<DenseLayer&>(layer2);
    auto layer3Copy = const_cast<DenseLayer&>(layer3);

    Matrix Z1 = layer1Copy.forward(X);
    Matrix A1 = tanhForward(Z1);

    Matrix Z2 = layer2Copy.forward(A1);
    Matrix A2 = tanhForward(Z2);

    Matrix logits = layer3Copy.forward(A2);

    int bestClass = 0;
    float bestValue = logits(0, 0);

    for (int c = 1; c < logits.cols; c++) {
        if (logits(0, c) > bestValue) {
            bestValue = logits(0, c);
            bestClass = c;
        }
    }

    return bestClass;
}

float NeuralNetworkFast::evaluate(const std::vector<TrainingSample>& testData) const {
    if (testData.empty()) {
        return 0.0f;
    }

    int correct = 0;
    for (const auto& sample : testData) {
        int pred = predict_digit(sample.features);
        if (pred == sample.label) {
            correct++;
        }
    }

    return static_cast<float>(correct) / static_cast<float>(testData.size());
}

bool NeuralNetworkFast::save_model(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    return true;
}

bool NeuralNetworkFast::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    return true;
}
