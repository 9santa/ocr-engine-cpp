#include "data/mnist_loader.h"
#include "baselines/common/training_sample.h"
#include "baselines/knn/feature_extractor.h"
#include "baselines/nn_mlp_fast/neural_network_fast.h"
#include "experiments/training_logger.h"

#include <iostream>
#include <vector>
#include <sstream>

int main() {
    MNISTLoader loader;
    if (!loader.loadTrainingData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")) {
        std::cerr << "failed to load train data\n";
        return 1;
    }

    if (!loader.loadTestData("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")) {
        std::cerr << "failed to load test data\n";
        return 1;
    }

    FeatureExtractor fe;

    std::vector<TrainingSample> trainSamples;
    std::vector<TrainingSample> testSamples;

    const auto& trainData = loader.getTrainingData();
    const auto& testData = loader.getTestData();

    std::vector<RunConfig> runs = {
        {"fastnn_t5000_v500_lr0.02_b64_e10", 5000, 500, 0.02f, 64, 10},
        {"fastnn_t5000_v500_lr0.01_b64_e10", 5000, 500, 0.01f, 64, 10},
        {"fastnn_t5000_v500_lr0.03_b64_e10", 5000, 500, 0.03f, 64, 10},
        {"fastnn_t5000_v500_lr0.02_b128_e20", 5000, 500, 0.02f, 128, 20},
        {"fastnn_t5000_v500_lr0.04_b128_e10", 5000, 500, 0.04f, 128, 10},
        {"fastnn_t5000_v500_lr0.02_b32_e10", 5000, 500, 0.02f, 32, 10}
    };

    // const int trainLimit = 5000;
    // const int testLimit = 500;
    // const float learningRate = 0.02f;
    // const int batchSize = 64;
    // const int epochs = 10;

    // std::ostringstream runNameBuilder;
    // runNameBuilder << "fastnn_t" << trainLimit
    //     << "_v" << testLimit
    //     << "_lr" << learningRate
    //     << "_b" << batchSize
    //     << "_e" << epochs;
    //
    // const std::string runName = runNameBuilder.str();

    for (const auto& run : runs) {
        const int trainLimit = run.trainLimit;
        const int testLimit = run.testLimit;
        const float learningRate = run.learningRate;
        const int batchSize = run.batchSize;
        const int epochs = run.epochs;
        const std::string runName = run.runName;
        trainSamples.reserve(trainLimit);
        testSamples.reserve(testLimit);

        for (size_t i = 0; i < trainLimit; i++) {
            TrainingSample s;
            s.features = fe.extractNeuralNetworkFeatures(trainData[i].image);
            s.label = trainData[i].label;
            trainSamples.push_back(std::move(s));
        }

        for (size_t i = 0; i < testLimit; i++) {
            TrainingSample s;
            s.features = fe.extractNeuralNetworkFeatures(testData[i].image);
            s.label = testData[i].label;
            testSamples.push_back(std::move(s));
        }



        RunConfig config;
        config.runName = runName;
        config.trainLimit = trainLimit;
        config.testLimit = testLimit;
        config.learningRate = learningRate;
        config.batchSize = batchSize;
        config.epochs = epochs;

        NeuralNetworkFast nn(784, 128, 64, 10);
        nn.train(trainSamples, epochs, learningRate, batchSize);

        float acc = nn.evaluate(testSamples);
        std::cout << "Fast NN accuracy: " << acc * 100.0f << "%\n";

        TrainingLogger::logSummary(config, acc);

    }

    return 0;
}
