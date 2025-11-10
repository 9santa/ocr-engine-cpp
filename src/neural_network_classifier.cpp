#include "../include/neural_network_classifier.h"
#include "gradient.h"
#include "image_matrix.h"
#include <memory>
#include <vector>
#include <iostream>


NeuralNetwork::NeuralNetwork(const std::vector<int>& layers)
    : layerSizes(layers) {
        network = std::make_unique<MLP>(layers[0], std::vector<int>(layers.begin()+1, layers.end()));
    }

std::vector<ValPtr> NeuralNetwork::featuresToValPtr(const std::vector<float>& features) {
    std::vector<ValPtr> result;
    for (float f : features) {
        result.push_back(std::make_shared<Value>(f));
    }
    return result;
}

std::vector<ValPtr> NeuralNetwork::one_hot(int label, int num_classes) {
    std::vector<ValPtr> vec(num_classes);
    for (int i = 0; i < num_classes; i++) {
        vec[i] = std::make_shared<Value>(i == label ? 1.0 : 0.0);
    }
    return vec;
}

void NeuralNetwork::train(const std::vector<TrainingSample>& trainingData, int epochs, float learningRate) {
    if (trainingData.empty()) return;

    // DEBUG: Use very small dataset for testing
    int debugSamples = std::min(50, static_cast<int>(trainingData.size()));
    std::cout << "DEBUG: Training on " << debugSamples << " samples only\n";

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epochLoss = 0.0f;

        for (int i = 0; i < debugSamples; i++) {
            // Zero gradients
            for (auto& param : network->parameters()) {
                param->grad = 0.0;
            }

            // Convert this sample
            auto input = featuresToValPtr(trainingData[i].features);
            auto target = one_hot(trainingData[i].label, layerSizes.back());
            auto prediction = (*network)(input);

            // Loss for this single sample
            ValPtr loss = std::make_shared<Value>(0.0);
            for (size_t j = 0; j < prediction.size(); j++) {
                auto diff = *prediction[j] - target[j];
                loss = *loss + *diff * diff;
            }
            loss = *loss * std::make_shared<Value>(1.0 / prediction.size());

            // Backward pass
            loss->backward();

            // Update parameters
            for (auto& param : network->parameters()) {
                param->data -= learningRate * param->grad;
            }

            epochLoss += loss->data;

            std::cout << "\rEpoch " << epoch << " - Sample " << i+1 << "/" << debugSamples 
                      << " - Loss: " << loss->data;
            std::cout.flush();

            loss->prev.clear();
            loss->_backward = []() {};

            for (auto& p : prediction) {
                p->prev.clear();
                p->_backward = []() {};
            }
            
            for (auto& t : target) {
                t->prev.clear(); 
                t->_backward = []() {};
            }
            
            for (auto& in : input) {
                in->prev.clear();
                in->_backward = []() {};
            }

        }

        // cleanup after each epoch
        for (auto& param : network->parameters()) {
            param->prev.clear();
            param->_backward = []() {};
        }

        std::cout << "\nEpoch " << epoch << " completed - Avg Loss: " 
                  << (epochLoss / debugSamples) << "\n";


    }

    std::cout << "Neural Network training completed without memory issues!\n";
}

#if 0
void NeuralNetwork::train(const std::vector<TrainingSample>& trainingData, int epochs, float learningRate) {
    if (trainingData.empty()) return;

    std::cout << "Training Neural Network (" << layerSizes[0] << "->";
    for (size_t i = 1; i < layerSizes.size(); i++) {
        std::cout << layerSizes[i] << (i == layerSizes.size() - 1 ? ")" : "->");
    }
    std::cout << " on " << trainingData.size() << " samples...\n";

    // Convert training data to neural network format (ValPtr)
    std::vector<std::vector<ValPtr>> inputs;
    std::vector<std::vector<ValPtr>> targets;

    for (const auto& sample : trainingData) {
        inputs.push_back(featuresToValPtr(sample.features));
        targets.push_back(one_hot(sample.label, layerSizes.back()));
    }

    int batchSize = 8; // mini-batch
    int totalBatches = (trainingData.size() + batchSize + 1) / batchSize;


    // Training loop
    for (int epoch = 0; epoch < 1; epoch++) {
        float epochLoss = 0.0f;
        int batchesProcessed = 0;

        for (int batchStart = 0; batchStart < trainingData.size(); batchStart += batchSize) {
            int batchEnd = std::min(batchStart + batchSize, static_cast<int>(trainingData.size()));
            int currentBatchSize = batchEnd - batchStart;

            // Zero gradients
            for (auto& param : network->parameters()) {
                param->grad = 0.0;
            }

            ValPtr batchLoss = std::make_shared<Value>(0.0);
            int samplesInBatch = 0;

            // Process one batch
            for (int i = batchStart; i < batchEnd; i++) {
                auto input = featuresToValPtr(trainingData[i].features);
                auto target = one_hot(trainingData[i].label, layerSizes.back());
                auto prediction = (*network)(input);

                // MSE loss for this sample
                ValPtr sampleLoss = std::make_shared<Value>(0.0);
                for (size_t j = 0; j < prediction.size(); j++) {
                    auto diff = *prediction[j] - target[j];
                    auto sq = *diff * diff;
                    sampleLoss = *sampleLoss + sq;
                }
                sampleLoss = *sampleLoss * std::make_shared<Value>(1.0 / prediction.size());

                batchLoss = *batchLoss + sampleLoss;
            }

            // Average loss for the batch
            batchLoss = *batchLoss * std::make_shared<Value>(1.0 / currentBatchSize);

            // Backward pass
            batchLoss->backward();

            // Update parameters
            for (auto& param : network->parameters()) {
                param->data -= learningRate * param->grad;
            }

            epochLoss += batchLoss->data;
            batchesProcessed++;

            // Progress within epoch
            if (batchesProcessed % 10 == 0) {
                std::cout << "\rEpoch " << epoch << " - Batch " << batchesProcessed << "/" << totalBatches 
                          << " - Avg Loss: " << (epochLoss / batchesProcessed);
                std::cout.flush();
            }
        }

        if (batchesProcessed > 0)
        std::cout << "\rEpoch " << epoch << "/" << epochs << " completed - Avg Loss: " 
                  << (epochLoss / batchesProcessed) << "\n";
    }

        std::cout << "Neural Network training completed!\n";
}
#endif

int NeuralNetwork::predict_digit(const std::vector<float>& features) {
    auto input = featuresToValPtr(features);
    auto output = (*network)(input);

    int predictedClass = 0;
    float maxOutput = output[0]->data;
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i]->data > maxOutput) {
            maxOutput = output[i]->data;
            predictedClass = i;
        }
    }
    return predictedClass;
}

float NeuralNetwork::evaluate(const std::vector<TrainingSample>& testData) {
    if (testData.empty()) return 0.0f;

    std::cout << "Evaluating Neural Network on " << testData.size() << " samples...\n";

    int correct = 0;
    int processed = 0;
    int progressInterval = testData.size() / 20;

    for (const auto& sample : testData) {
        int predicition = predict_digit(sample.features);
        if (predicition == sample.label) correct++;
        processed++;

        // Progress indicator
        if (processed % progressInterval == 0) {
            float progress = static_cast<float>(processed) / testData.size() * 100.0f;
            std::cout << "\rProgress: " << static_cast<int>(progress) << "%" << std::flush;
        }
    }

    std::cout << "\n Neural Network Evaluation Completed!\n";
    return static_cast<float>(correct) / testData.size();
}
