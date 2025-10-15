#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

struct TrainingSample {
    std::vector<float> features;
    int label;
};

class KNNClassifier {
public:
    KNNClassifier(int k = 3);
    void train(const std::vector<TrainingSample>& trainingData);
    int predict(const std::vector<float>& features) const;
    float evaluate(const std::vector<TrainingSample>& testData) const;

private:
    int k;
    std::vector<TrainingSample> trainingData;

    float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    std::vector<std::pair<int, float>> findKNearest(const std::vector<float>& features) const;
};

#endif // KNN_CLASSIFIER_H
