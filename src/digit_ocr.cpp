#include "../include/digit_ocr.h"
#include "../include/mnist_loader.h"
#include <fstream>
#include <iostream>

DigitOCR::DigitOCR() : classifier(3) {}

std::vector<TrainingSample> DigitOCR::loadMNISTData(const std::string& imagePath, const std::string& lapelPath) {
    std::vector<TrainingSample> samples;

    return samples;
}

void DigitOCR::trainModel(const std::string& trainingDataPath) {
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

    std::cout << "Extracting features from " << mnistData.size() << " training images...\n";

    for (const auto& mnistImage : mnistData) {
        TrainingSample sample;
        sample.features = FeatureExtractor.extractFeatures(mnistImage.image);
        sample.label = mnistImage.label;
        trainingSamples.push_back(sample);
    }

    // train the classifier
    classifier.train(trainingSamples);
    std::cout << "Training completed with " << trainingSamples.size() << " samples\n";

    // evaluate on traning data
    float accuracy = classifier.evaluate(trainingSamples);
    std::cout << "Training accuracy: " << accuracy * 100 << "%\n";

}

std::string DigitOCR::recognize(const ImageMatrix& image) {
    auto digits = preprocessor.extractDigits(image);
    std::string result;

    for (const auto& digit : digits) {
        auto features = FeatureExtractor.extractFeatures(digit);
        int prediction = classifier.predict(features);
        result += std::to_string(prediction);
    }

    return result;
}

// model persistence
void DigitOCR::saveModel(const std::string& filename) {
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

void DigitOCR::loadModel(const std::string& filename) {
    std::cout << "Loading model from " << filename << "\n";
}
