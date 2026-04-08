#include "app/digit_ocr.h"

#include <fstream>
#include <iostream>
#include <utility>

DigitOCR::DigitOCR() : classifier(3), nnClassifier({784, 128, 64, 10}) {}

std::vector<TrainingSample> DigitOCR::loadMNISTSamples(
    const std::vector<MNISTImage>& data,
    AlgorithmType algo) const {
    std::vector<TrainingSample> samples;
    samples.reserve(data.size());

    for (const auto& item : data) {
        TrainingSample sample;
        sample.features = (algo == AlgorithmType::KNN)
            ? featureExtractor.extractKNNFeatures(item.image)
            : featureExtractor.extractNeuralNetworkFeatures(item.image);
        sample.label = item.label;
        samples.push_back(std::move(sample));
    }

    return samples;
}

void DigitOCR::trainModel(const std::string& trainingDataPath, AlgorithmType algo) {
    MNISTLoader loader;

    const std::string trainImages = trainingDataPath + "/train-images-idx3-ubyte";
    const std::string trainLabels = trainingDataPath + "/train-labels-idx1-ubyte";

    if (!loader.loadTrainingData(trainImages, trainLabels)) {
        std::cerr << "Failed to load training data!\n";
        return;
    }

    const auto& mnistData = loader.getTrainingData();
    std::cout << "Extracting features from " << mnistData.size() << " training images...\n";

    if (algo == AlgorithmType::KNN) {
        knnTrainingSamples = loadMNISTSamples(mnistData, algo);

        std::cout << "Training KNN classifier...\n";
        classifier.train(&knnTrainingSamples);
        knnTrained = true;

        std::cout << "KNN training completed with " << knnTrainingSamples.size() << " samples\n";
        return;
    }

    std::vector<TrainingSample> neuralSamples = loadMNISTSamples(mnistData, algo);

    neuralSamples.resize(1000);

    std::cout << "Training Neural Network...\n";
    nnClassifier.train(neuralSamples, 20, 0.075f);
    neuralNetworkTrained = true;

    std::cout << "Neural Network training completed with " << neuralSamples.size() << " samples\n";
    nnClassifier.save_model("digit_model.bin");
}

std::string DigitOCR::recognize(const ImageMatrix& image, AlgorithmType algo) {
    const auto digits = preprocessor.extractDigits(image);
    std::string result;

    for (const auto& digit : digits) {
        const auto features = (algo == AlgorithmType::KNN)
            ? featureExtractor.extractKNNFeatures(digit)
            : featureExtractor.extractNeuralNetworkFeatures(digit);

        const int prediction = (algo == AlgorithmType::KNN)
            ? classifier.predict(features)
            : nnClassifier.predict_digit(features);
        result += std::to_string(prediction);
    }

    return result;
}

void DigitOCR::loadModel(const std::string& filename, AlgorithmType algo) {
    if (algo == AlgorithmType::NN_SCALAR_AUTODIFF) {
        if (!nnClassifier.load_model(filename)) {
            std::cerr << "Cannot load neural-network model from " << filename << "\n";
        } else {
            neuralNetworkTrained = true;
        }
        return;
    }

    std::cout << "Loading model from " << filename << "\n";

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot load model from " << filename << "\n";
        return;
    }

    knnTrainingSamples.clear();
    size_t sampleCount = 0;
    file.read(reinterpret_cast<char*>(&sampleCount), sizeof(sampleCount));

    for (size_t i = 0; i < sampleCount; i++) {
        TrainingSample sample;

        file.read(reinterpret_cast<char*>(&sample.label), sizeof(sample.label));

        size_t featuresCount = 0;
        file.read(reinterpret_cast<char*>(&featuresCount), sizeof(featuresCount));
        sample.features.resize(featuresCount);
        file.read(reinterpret_cast<char*>(sample.features.data()), featuresCount * sizeof(float));

        knnTrainingSamples.push_back(std::move(sample));
    }

    classifier.train(&knnTrainingSamples);
    knnTrained = true;
    std::cout << "Model loaded from " << filename << " with " << sampleCount << " samples\n";
}

float DigitOCR::evaluateOnTestData(const std::string& testDataPath, AlgorithmType algo) {
    if (!isTrained(algo)) {
        std::cerr << "Error: Model is not trained yet!\n";
        return 0.0f;
    }

    MNISTLoader loader;
    const std::string testImages = testDataPath + "/t10k-images-idx3-ubyte";
    const std::string testLabels = testDataPath + "/t10k-labels-idx1-ubyte";

    if (!loader.loadTestData(testImages, testLabels)) {
        std::cerr << "Failed to load test data!\n";
        return 0.0f;
    }

    const auto& testData = loader.getTestData();
    std::cout << "Evaluating on " << testData.size() << " test samples...\n";

    auto testSamples = loadMNISTSamples(testData, algo);
    if (algo == AlgorithmType::KNN) {
        const float accuracy = classifier.evaluate(testSamples, 500, true);
        std::cout << "KNN Test Accuracy: " << accuracy * 100 << "%\n";
        return accuracy;
    }

    const float accuracy = nnClassifier.evaluate(testSamples);
    std::cout << "Neural Network Test Accuracy: " << accuracy * 100 << "%\n";
    return accuracy;
}

float DigitOCR::evaluateOnTrainingData() {
    if (knnTrainingSamples.empty()) {
        return 0.0f;
    }

    return classifier.evaluate(knnTrainingSamples);
}

void DigitOCR::confusionMatrix(const std::vector<MNISTImage>&, AlgorithmType) {}

void DigitOCR::saveModel(const std::string& filename, AlgorithmType algo) {
    if (algo == AlgorithmType::NN_SCALAR_AUTODIFF) {
        nnClassifier.save_model(filename);
        return;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot save model to " << filename << "\n";
        return;
    }

    const size_t sampleCount = knnTrainingSamples.size();
    file.write(reinterpret_cast<const char*>(&sampleCount), sizeof(sampleCount));

    for (const auto& sample : knnTrainingSamples) {
        file.write(reinterpret_cast<const char*>(&sample.label), sizeof(sample.label));

        const size_t featureCount = sample.features.size();
        file.write(reinterpret_cast<const char*>(&featureCount), sizeof(featureCount));
        file.write(
            reinterpret_cast<const char*>(sample.features.data()),
            static_cast<std::streamsize>(featureCount * sizeof(float)));
    }
}

bool DigitOCR::isTrained(AlgorithmType algo) const {
    if (algo == AlgorithmType::KNN) {
        return knnTrained;
    }

    return neuralNetworkTrained;
}
