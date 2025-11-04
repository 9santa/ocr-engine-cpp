#ifndef DIGIT_OCR_H
#define DIGIT_OCR_H

#include "feature_extractor.h"
#include "knn_classifier.h"
#include "image_matrix.h"
#include <vector>
#include <string>

class DigitOCR {
public:
    DigitOCR();

    void trainModel(const std::string& trainingDataPath);
    std::string recognize(const ImageMatrix& image);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

private:
    FeatureExtractor FeatureExtractor;
    KNNClassifier classifier;
    std::vector<TrainingSample> trainingSamples;

    std::vector<TrainingSample> loadMNISTData(const std::string& imagePath, const std::string& labelPath);

};


#endif // !DIGIT_OCR_H
