#include "knn_classifier.h"


KNNClassifier::KNNClassifier(int k) : k(k) {}

void KNNClassifier::train(const std::vector<TrainingSample>& trainingData) {
    this->trainingData = trainingData;
}

float KNNClassifier::euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) const {
    float distance = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

std::vector<std::pair<int, float>> KNNClassifier::findKNearest(const std::vector<float>& features) const {
    std::vector<std::pair<int, float>> distances;

    for (const auto& sample : trainingData) {
        float distance = euclideanDistance(features, sample.features);
        distances.emplace_back(sample.label, distance);
    }

    // Sort by distance
    std::sort(distances.begin(), distances.end(), 
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

    // Return k nearest neighbors
    if (static_cast<int>(distances.size()) > k) {
        distances.resize(k);
    }

    return distances;
}

int KNNClassifier::predict(const std::vector<float>& features) const {
    auto neighbors = findKNearest(features);

    // Count votes
    std::map<int, int> votes;
    for (const auto& neighbor : neighbors) {
        votes[neighbor.first]++;
    }

    // Find label with most votes
    int predictedLabel = -1;
    int maxVotes = 0;
    for (const auto& vote : votes) {
        if (vote.second > maxVotes) {
            maxVotes = vote.second;
            predictedLabel = vote.first;
        }
    }

    return predictedLabel;
}

float KNNClassifier::evaluate(const std::vector<TrainingSample>& testData) const {
    int correct = 0;
    
    for (const auto& sample : testData) {
        int prediction = predict(sample.features);
        if (prediction == sample.label) correct++;
    }

    return static_cast<float>(correct) / testData.size();
}
