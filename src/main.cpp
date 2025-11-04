#include "../include/digit_ocr.h"
#include "../include/bmp_reader.h"
#include <iostream>

signed main(void) {
    DigitOCR ocr;
    
    // train or load model
    std::cout << "1. Train new model\n2. Load existing model\nChoose: ";
    unsigned short choice;
    std::cin >> choice;

    if (choice == 1) {
        std::string dataPath;
        std::cout << "Enter path to MNIST data folder: ";
        std::cin >> dataPath;
        ocr.trainModel(dataPath);
        ocr.saveModel("trained_model.dat");
    } else {
        ocr.loadModel("trained_model.dat");
    }

    return 0;
}
