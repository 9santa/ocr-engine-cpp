#include "../include/mnist_loader.h"
#include <fstream>
#include <ios>
#include <iostream>

MNISTLoader::MNISTLoader() {}

int MNISTLoader::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool MNISTLoader::loadImages(const std::string& imagePath, std::vector<MNISTImage>& data) {
    std::ifstream file(imagePath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Cannot open image file: " << imagePath << "\n";
        return false;
    }

    int magicNumber = 0;
    int numberOfImages = 0;
    int nRows;
    int nCols;

    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    file.read((char*)&numberOfImages, sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);

    file.read((char*)&nRows, sizeof(nRows));
    nRows = reverseInt(nRows);

    file.read((char*)&nCols, sizeof(nCols));
    nCols = reverseInt(nCols);

    std::cout << "Loading " << numberOfImages << " images of size " << nRows << "x" << nCols << "\n";

    data.resize(numberOfImages);

    for (int i = 0; i < numberOfImages; i++) {
        ImageMatrix image(nCols, nRows, 1);
        for (int r = 0; r < nRows; r++) {
            for (int c = 0; c < nCols; c++) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, sizeof(pixel));
                image(r, c, 0) = pixel;
            }
        }
        data[i].image = image;
    }

    file.close();
    return true;
}

bool MNISTLoader::loadLabels(const std::string& labelPath, std::vector<MNISTImage>& data) {
    std::ifstream file(labelPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open label file: " << labelPath << "\n";
        return false;
    }

    int magicNumber;
    int numberOfLabels;

    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    file.read((char*)&numberOfLabels, sizeof(numberOfLabels));
    numberOfLabels = reverseInt(numberOfLabels);

    if (numberOfLabels != data.size()) {
        std::cerr << "Number of labels doesn't match number of images!\n";
        return false;
    }

    for (int i = 0; i < numberOfLabels; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        data[i].label = label;
    }

    file.close();
    return true;
}

bool MNISTLoader::loadTrainingData(const std::string& imagePath, const std::string& labelPath) {
    if (!loadImages(imagePath, trainingData)) return false;
    if (!loadLabels(labelPath, trainingData)) return false;

    std::cout << "Successfuly loaded " << trainingData.size() << " training samples\n";
    return true;
}

bool MNISTLoader::loadTestData(const std::string& imagePath, const std::string& labelPath) {
    if (!loadImages(imagePath, testData)) return false;
    if (!loadLabels(labelPath, testData)) return false;

    std::cout << "Successfuly loaded " << trainingData.size() << " test samples\n";
    return true;
}

std::vector<MNISTImage> MNISTLoader::getTrainingData() const {
    return trainingData;
}

std::vector<MNISTImage> MNISTLoader::getTestData() const {
    return testData;
}
