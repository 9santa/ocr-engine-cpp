#pragma once
#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include "baselines/common/training_sample.h"
#include <vector>

class KNNClassifier {
public:
    KNNClassifier(int k = 3);
    void train(const std::vector<TrainingSample>* trainingData);
    int predict(const std::vector<float>& features) const;
    float evaluate(const std::vector<TrainingSample>& testData) const;

private:
    int k;
    const std::vector<TrainingSample>* trainingData = nullptr;

    float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) const;

    std::vector<std::pair<int, float>> findKNearest(const std::vector<float>& features) const;
};

#endif // KNN_CLASSIFIER_H
