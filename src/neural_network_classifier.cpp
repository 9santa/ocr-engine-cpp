#include "neural_network_classifier.h"
#include "loss.h"
#include "opt_value.h"
#include "opt_ops.h"
#include <vector>
#include <iostream>


NeuralNetwork::NeuralNetwork(const std::vector<int>& layers)
    : layerSizes(layers), network(layers[0], std::vector<int>(layers.begin()+1, layers.end())) {}

// Features -> OptValPtr
std::vector<OptValPtr> NeuralNetwork::featuresToOptValPtr(const std::vector<float>& features) {
    std::vector<OptValPtr> result;
    for (float f : features) {
        result.push_back(OptValue::make_value(f));
    }
    return result;
}

// One-hot -> OptValPtr
std::vector<OptValPtr> one_hot_opt(int label, int num_classes) {
    std::vector<OptValPtr> vec(num_classes);
    for (int i = 0; i < num_classes; i++) {
        vec[i] = OptValue::make_value(i == label ? 1.0 : 0.0);
    }
    return vec;
}

void NeuralNetwork::train(const std::vector<TrainingSample>& trainingData, int epochs, float learningRate) {
    if (trainingData.empty()) return;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epochLoss = 0.0f;

        for (size_t i = 0; i < trainingData.size(); i++) {
            // Zero gradients
            network.zero_grad();

            // Convert this sample
            auto input = featuresToOptValPtr(trainingData[i].features);
            auto target = trainingData[i].label;
            auto prediction = network(input);

            auto loss = cross_entropy_loss(prediction, target);
            loss->grad = 1.0;
            apply_gradients();

            for (auto& param : network.parameters()) {
                param->data -= learningRate * param->grad;
            }

            epochLoss += loss->data;

            // if ((i+1) % 100 == 0 || i+1 == trainingData.size()) {
            //     std::cout << "\rEpoch " << epoch << " - Sample " << i+1 << "/" << trainingData.size()
            //               << " - Loss: " << loss->data;
            //     std::cout.flush();
            // }
        }

        std::cout << "\nEpoch " << epoch << " completed - Avg Loss: "
                  << (epochLoss / trainingData.size()) << "\n";
    }

    network.zero_grad();
    std::cout << "Neural Network training completed!\n";
}

// Preciction
int NeuralNetwork::predict_digit(const std::vector<float>& features) {
    auto input = featuresToOptValPtr(features);
    auto output = network(input);

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

// Evaluation
float NeuralNetwork::evaluate(std::vector<TrainingSample>& testData) {
    if (testData.empty()) return 0.0f;

    testData.resize(100);

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



#if 0
OptMLP network_opt(784, {128, 64, 10});

void NeuralNetwork::train_opt(const std::vector<TrainingSample>& trainingData, int epochs, float learningRate) {

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        int processed = 0;

        for (const auto& sample : trainingData) {
            gradient_ops.clear();
            network_opt.zero_grad();

            auto input = featuresToOptValPtr(sample.features);
            auto target = one_hot_opt(sample.label, 10);
            auto prediction = network_opt(input);
            auto loss = mse_opt(target, prediction);

            loss->grad = 1.0;
            OptValue::apply_gradients();

            for (auto& param : network_opt.parameters()) {
                param->data -= learningRate * param->grad;
            }

            epoch_loss += loss->data;
            processed++;

            if (processed % 100 == 0) {
                std::cout << "\rEpoch " << epoch << " - " << processed 
                          << " samples - Loss: " << loss->data;
                std::cout.flush();
            }
        }

        std::cout << "\nEpoch " << epoch << " completed - Avg Loss: " 
                  << (epoch_loss / processed) << "\n";
    }
}


// std::vector<ValPtr> NeuralNetwork::featuresToValPtr(const std::vector<float>& features) {
//     std::vector<ValPtr> result;
//     for (float f : features) {
//         result.push_back(std::make_shared<Value>(f));
//     }
//     return result;
// }
//
// std::vector<ValPtr> NeuralNetwork::one_hot(int label, int num_classes) {
//     std::vector<ValPtr> vec(num_classes);
//     for (int i = 0; i < num_classes; i++) {
//         vec[i] = std::make_shared<Value>(i == label ? 1.0 : 0.0);
//     }
//     return vec;
// }

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
