#pragma once
#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "image_matrix.h"
#include <vector>
#include <string>

struct MNISTImage {
    ImageMatrix image;
    int label;
};

class MNISTLoader {
public:
    MNISTLoader();
    bool loadTrainingData(const std::string& imagePath, const std::string& labelPath);
    bool loadTestData(const std::string& imagePath, const std::string& labelPath);
    std::vector<MNISTImage> getTrainingData() const;
    std::vector<MNISTImage> getTestData() const;

private:
    std::vector<MNISTImage> trainingData;
    std::vector<MNISTImage> testData;

    int reverseInt(int i);
    bool loadImages(const std::string& imagePath, std::vector<MNISTImage>& data);
    bool loadLabels(const std::string& labelPath, std::vector<MNISTImage>& data);
};


#endif // !MNIST_LOADER_H
