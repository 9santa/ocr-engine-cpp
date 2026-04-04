#include "baselines/neural_network/neural_network_classifier.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "baselines/neural_network/nn/loss.h"
#include "baselines/neural_network/nn/opt_ops.h"
#include "baselines/neural_network/nn/opt_value.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers)
    : layerSizes(layers), network(layers[0], std::vector<int>(layers.begin() + 1, layers.end())) {}

std::vector<OptValPtr> NeuralNetwork::featuresToOptValPtr(
    GraphArena& arena,
    const std::vector<float>& features) {
    std::vector<OptValPtr> result;
    result.reserve(features.size());
    for (float feature : features) {
        result.push_back(arena.make_value(static_cast<double>(feature)));
    }
    return result;
}

std::vector<OptValPtr> NeuralNetwork::one_hot(GraphArena& arena, int label, int num_classes) {
    std::vector<OptValPtr> values(num_classes);
    for (int i = 0; i < num_classes; i++) {
        values[i] = arena.make_value(i == label ? 1.0 : 0.0);
    }
    return values;
}

void NeuralNetwork::train(
    const std::vector<TrainingSample>& trainingData,
    int epochs,
    float learningRate) {
    if (trainingData.empty()) {
        return;
    }

    const size_t totalSamples = trainingData.size();

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epochLoss = 0.0f;
        size_t samplesProcessed = 0;

        for (const auto& sample : trainingData) {
            GraphArena arena(300000);
            tape().clear();
            network.zero_grad();

            const auto input = featuresToOptValPtr(arena, sample.features);
            const auto prediction = network(arena, input);
            auto loss = cross_entropy_loss(arena, prediction, sample.label);
            loss->grad = 1.0;
            apply_gradients();

            for (auto& param : network.parameters()) {
                param->data -= learningRate * param->grad;
            }

            epochLoss += static_cast<float>(loss->data);
            ++samplesProcessed;

            if (samplesProcessed % 1000 == 0 || samplesProcessed == totalSamples) {
                std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs
                          << " - Sample " << samplesProcessed << "/" << totalSamples
                          << " - Avg Loss: " << (epochLoss / samplesProcessed);
                std::cout.flush();
            }
        }

        std::cout << "\nEpoch " << epoch + 1 << " completed - Avg Loss: "
                  << (epochLoss / trainingData.size()) << "\n";
    }

    network.zero_grad();
    std::cout << "Neural Network training completed!\n";
}

int NeuralNetwork::predict_digit(const std::vector<float>& features) {
    GraphArena arena;
    tape().clear();

    const auto input = featuresToOptValPtr(arena, features);
    const auto output = network(arena, input);

    int predictedClass = 0;
    double maxOutput = output[0]->data;
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i]->data > maxOutput) {
            maxOutput = output[i]->data;
            predictedClass = static_cast<int>(i);
        }
    }
    return predictedClass;
}

float NeuralNetwork::evaluate(std::vector<TrainingSample>& testData) {
    if (testData.empty()) {
        return 0.0f;
    }

    std::cout << "Evaluating Neural Network on " << testData.size() << " samples...\n";

    int correct = 0;
    int processed = 0;
    const int progressInterval = std::max(1, static_cast<int>(testData.size() / 20));

    for (const auto& sample : testData) {
        const int prediction = predict_digit(sample.features);
        if (prediction == sample.label) {
            correct++;
        }
        processed++;

        if (processed % progressInterval == 0) {
            const float progress = static_cast<float>(processed) / testData.size() * 100.0f;
            std::cout << "\rProgress: " << static_cast<int>(progress) << "%" << std::flush;
        }
    }

    std::cout << "\nNeural Network Evaluation Completed!\n";
    return static_cast<float>(correct) / testData.size();
}

bool NeuralNetwork::save_model(const std::string& filename) const {
    std::cout << "Saving model to " << filename << "\n";

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot save model to " << filename << "\n";
        return false;
    }

    const int numLayers = static_cast<int>(layerSizes.size());
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
    file.write(reinterpret_cast<const char*>(layerSizes.data()), numLayers * sizeof(int));

    for (const auto& layer : network.layers) {
        for (const auto& neuron : layer.neurons) {
            for (auto* weight : neuron.w) {
                const double value = weight->data;
                file.write(reinterpret_cast<const char*>(&value), sizeof(double));
            }

            const double bias = neuron.b->data;
            file.write(reinterpret_cast<const char*>(&bias), sizeof(double));
        }
    }

    return true;
}

bool NeuralNetwork::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open file for loading: " << filename << "\n";
        return false;
    }

    int numLayers = 0;
    in.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
    if (!in || numLayers <= 0) {
        std::cerr << "Invalid model file (numLayers)\n";
        return false;
    }

    std::vector<int> sizes(numLayers);
    in.read(reinterpret_cast<char*>(sizes.data()), numLayers * sizeof(int));
    if (!in) {
        std::cerr << "Invalid model file (layerSizes)\n";
        return false;
    }

    if (sizes != layerSizes) {
        layerSizes = sizes;
        network = OptMLP(layerSizes[0], std::vector<int>(layerSizes.begin() + 1, layerSizes.end()));
    }

    for (auto& layer : network.layers) {
        for (auto& neuron : layer.neurons) {
            for (auto* weight : neuron.w) {
                double value = 0.0;
                in.read(reinterpret_cast<char*>(&value), sizeof(double));
                if (!in) {
                    std::cerr << "Unexpected end of file while reading weights\n";
                    return false;
                }
                weight->data = value;
            }

            double bias = 0.0;
            in.read(reinterpret_cast<char*>(&bias), sizeof(double));
            if (!in) {
                std::cerr << "Unexpected end of file while reading bias\n";
                return false;
            }
            neuron.b->data = bias;
        }
    }

    return true;
}
