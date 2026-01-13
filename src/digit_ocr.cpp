#include "digit_ocr.h"
#include "mnist_loader.h"
#include "preprocessor.h"
#include "knn_classifier.h"
#include <fstream>
#include <iostream>

DigitOCR::DigitOCR() : classifier(3), nnClassifier({784, 128, 64, 10}) {}

std::vector<TrainingSample> DigitOCR::loadMNISTData(const std::string& imagePath, const std::string& lapelPath) {
    std::vector<TrainingSample> samples;

    return samples;
}

void DigitOCR::trainModel(const std::string& trainingDataPath, AlgorithmType algo) {
    MNISTLoader loader;

    // load MNIST data
    std::string trainImages = trainingDataPath + "/train-images-idx3-ubyte";
    std::string trainLabels = trainingDataPath + "/train-labels-idx1-ubyte";

    if (!loader.loadTrainingData(trainImages, trainLabels)) {
        std::cerr << "Failed to load training data!\n";
        return;
    }

    // convert MNIST images to training samples
    auto mnistData = loader.getTrainingData();
    trainingSamples.clear();

    int maxSamplesForNN = (algo == AlgorithmType::NEURAL_NETWORK) ? 50 : mnistData.size();
    if (algo == AlgorithmType::NEURAL_NETWORK) mnistData.resize(maxSamplesForNN);

    std::cout << "Extracting features from " << mnistData.size() << " training images...\n";

    // Add progress bar
    int totalImages = mnistData.size();
    int progressInterval = totalImages / 20;    // update every 5%

    for (int i = 0; i < totalImages; i++) {
        const auto& mnistImage = mnistData[i];
        TrainingSample sample;
        sample.features = featureExtractor.extractFeatures(mnistImage.image);
        sample.label = mnistImage.label;
        trainingSamples.push_back(sample);

        // Show progress
        if (i % progressInterval == 0 || i == totalImages - 1) {
            float progress = static_cast<float>(i+1) / totalImages * 100.0f;
            std::cout << "\nProgress: " << static_cast<int>(progress) << "% (" << i + 1 << "/" << totalImages << ")" << std::flush;
        }
    }

    std::cout << "\n";

    // train with the selected algorithm type
    if (algo == AlgorithmType::KNN) {
        std::cout << "Training KNN classifier...\n";
        classifier.train(trainingSamples);
        std::cout << "KNN training completed with " << trainingSamples.size() << " samples\n";
    } else {
        std::cout << "Training Neural Network...\n";
        nnClassifier.train(trainingSamples, 20, 0.075f);  // 50 epochs, 0.01 learning rate
        std::cout << "Neural Network training completed with " << trainingSamples.size() << " samples\n";
    }
}

std::string DigitOCR::recognize(const ImageMatrix& image, AlgorithmType algo) {
    auto digits = preprocessor.extractDigits(image);
    std::string result;

    for (const auto& digit : digits) {
        auto features = featureExtractor.extractFeatures(digit);
        int prediction = classifier.predict(features);
        result += std::to_string(prediction);
    }

    return result;
}

// Saving model to a file
void DigitOCR::saveModel(const std::string& filename, AlgorithmType algo) {
    std::cout << "Saving model to " << filename << "\n";

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot save model to " << filename << "\n";
        return;
    }

    // save traning samples
    size_t sampleCount = trainingSamples.size();
    file.write((const char*)&sampleCount, sizeof(sampleCount));

    for (const auto& sample : trainingSamples) {
        // save label
        file.write((const char*)&sample.label, sizeof(sample.label));

        // save features
        size_t featuresCount = sample.features.size();
        file.write((const char*)&featuresCount, sizeof(featuresCount));
        file.write((const char*)(sample.features.data()), featuresCount * sizeof(float));
    }

    std::cout << "Model saved to " << filename << " with " << sampleCount << " samples\n";
}

// Loading model from a file
void DigitOCR::loadModel(const std::string& filename, AlgorithmType algo) {
    std::cout << "Loading model from " << filename << "\n";

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot load model from " << filename << "\n";
        return;
    }

    trainingSamples.clear();
    size_t sampleCount;
    file.read((char*)&sampleCount, sizeof(sampleCount));

    for (size_t i = 0; i < sampleCount; i++) {
        TrainingSample sample;

        // load label
        file.read((char*)&sample.label, sizeof(sample.label));

        // load features
        size_t featuresCount;
        file.read((char*)&featuresCount, sizeof(featuresCount));
        sample.features.resize(featuresCount);
        file.read((char*)sample.features.data(), featuresCount * sizeof(float));

        trainingSamples.push_back(sample);
    }

    classifier.train(trainingSamples);
    std::cout << "Model loaded from " << filename << " with " << sampleCount << " samples\n";
}

float DigitOCR::evaluateOnTestData(const std::string& testDataPath, AlgorithmType algo) {
    if (!isTrained()) {
        std::cerr << "Error: Model is not trained yet!\n";
        return 0.0f;
    }

    MNISTLoader loader;
    std::string testImages = testDataPath + "/t10k-images-idx3-ubyte";
    std::string testLabels = testDataPath + "/t10k-labels-idx1-ubyte";

    if (!loader.loadTestData(testImages, testLabels)) {
        std::cerr << "Failed to load test data!\n";
        return 0.0f;
    }

    auto testData = loader.getTestData();
    std::cout << "Evaluating on " << testData.size() << " test samples...\n";

    // Convert MNIST test data to my TrainingSample format
    std::vector<TrainingSample> testSamples;
    for (const auto& mnist : testData) {
        TrainingSample sample;
        sample.features = featureExtractor.extractFeatures(mnist.image);
        sample.label = mnist.label;
        testSamples.push_back(sample);
    }

    // Evaluate with KNN classifier
    float KNNaccuracy = classifier.evaluate(testSamples);

    std::cout << "KNN Test Accuracy: " << KNNaccuracy * 100 << "%\n";

    // Evaluate with Neural Network classifier
    float NNaccuracy = nnClassifier.evaluate(testSamples);

    std::cout << "Neural Network Test Accuracy: " << NNaccuracy * 100 << "%\n";

    return KNNaccuracy;
}
