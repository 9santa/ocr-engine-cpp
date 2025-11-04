#include "../include/knn_classifier.h"
#include <iostream>

signed main(void) {
    KNNClassifier knn(3);

    // simple training data
    std::vector<TrainingSample> trainingData = {
        {{0.1, 0.1}, 0},
        {{0.2, 0.2}, 0},
        {{0.8, 0.8}, 1},
        {{0.9, 0.9}, 1}
    };

    // train
    knn.train(trainingData);

    // test prediction
    std::vector<float> testFeatures = {0.15, 0.15};
    int prediction = knn.predict(testFeatures);

    std::cout << "Prediction: " << prediction << " (expected: 0)" << std::endl;

    // test evaluation
    float accuracy = knn.evaluate(trainingData);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;


    return 0;
}
