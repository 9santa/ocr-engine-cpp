#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "feature_extractor.h"
#include "knn_classifier.h"
#include <vector>

struct MNISTImage {
    cv::Mat image;
    int lavel;
};

class MNISTLoader {
public:
    MNISTLoader();
    bool loadTrainingData(const std::string& imagePath, const std::string& labelPath);
    bool loadTestData(const std::string& imagePath, const std::string& labelPath);
    std::vector<TrainingSample> getTrainingSamples() const;
    std::vector<TrainingSample> getTestSamples() const;
    void displaySample(int index, bool isTest = false) const;

private:
    std::vector<MNISTImage> trainingData;
    std::vector<MNISTImage> testData;
    FeatureExtractor feature_extractor;

    int reverseInt(int i);
    bool loadImages(const std::string& imagePath, std::vector<MNISTImage>& data);
    bool loadLabels(const std::string& labelPath, std::vector<MNISTImage>& data);
};


#endif // !MNIST_LOADER_H
