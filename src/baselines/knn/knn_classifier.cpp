#include "baselines/knn/knn_classifier.h"
#include <algorithm>
#include <map>
#include <iostream>
#include <limits>

KNNClassifier::KNNClassifier(int k) : k(k) {}

void KNNClassifier::train(const std::vector<TrainingSample>* trainingData) {
    this->trainingData = trainingData;
}

float KNNClassifier::euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) const {
    float distance = 0.0f;
    const std::size_t n = std::min(a.size(), b.size());

    for (size_t i = 0; i < n; i++) {
        const float diff = a[i] - b[i];
        distance += diff * diff;
    }

    // intentionally remove std::sqrt() for speed purposes
    return distance;
}

std::vector<std::pair<int, float>> KNNClassifier::findKNearest(const std::vector<float>& features) const {
    std::vector<std::pair<int, float>> distances;

    if (!trainingData || trainingData->empty()) {
        return distances;
    }

    distances.reserve(trainingData->size());

    for (const auto& sample : *trainingData) {
        const float distance = euclideanDistance(features, sample.features);
        distances.emplace_back(sample.label, distance);
    }

    if (static_cast<int>(distances.size()) <= k) {
        return distances;
    }

    // Sort by distance
    std::nth_element(
        distances.begin(),
        distances.begin() + k,
        distances.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

    distances.resize(k);
    return distances;
}

int KNNClassifier::predict(const std::vector<float>& features) const {
    const auto neighbors = findKNearest(features);

    if (neighbors.empty()) {
        return -1;
    }

    // Count votes + distance tie-break
    std::map<int, int> votes;
    std::map<int, float> distanceSums;

    for (const auto& [label, dist] : neighbors) {
        votes[label]++;
        distanceSums[label] += dist;
    }

    // Find label with most votes
    int predictedLabel = -1;
    int maxVotes = -1;
    float bestDistanceSum = std::numeric_limits<float>::max();

    for (const auto& [label, count] : votes) {
        const float distSum = distanceSums[label];

        if (count > maxVotes || (count == maxVotes && distSum < bestDistanceSum)) {
            maxVotes = count;
            bestDistanceSum = distSum;
            predictedLabel = label;
        }
    }

    return predictedLabel;
}

float KNNClassifier::evaluate(
    const std::vector<TrainingSample>& testData,
    std::size_t maxSamples,
    bool showProgress) const {

    if (!trainingData || trainingData->empty() || testData.empty()) {
        return 0.0f;
    }

    const std::size_t limit = (maxSamples == 0) ? testData.size() : std::min(maxSamples, testData.size());

    int correct = 0;
    const std::size_t progressStep = std::max<std::size_t>(1, limit / 20);

    for (std::size_t i = 0; i < limit; i++) {
        const int prediction = predict(testData[i].features);
        if (prediction == testData[i].label) {
            correct++;
        }

        if (showProgress && ((i + 1) % progressStep == 0 || i + 1 == limit)) {
            const float progress = 100.0f * static_cast<float>(i + 1) / static_cast<float>(limit);
            std::cout << "\rKNN eval progress: " << (i + 1) << "/" << limit
                      << " (" << static_cast<int>(progress) << "%)" << std::flush;
        }
    }

    if (showProgress) {
        std::cout << "\n";
    }

    return static_cast<float>(correct) / static_cast<float>(limit);
}
