#include "app/cli.h"

#include "app/digit_ocr.h"
#include "io/bmp_reader.h"

#include <iostream>
#include <string>

namespace {
const char* algorithmName(AlgorithmType algo) {
    return (algo == AlgorithmType::KNN) ? "KNN" : "Neural Network";
}
} // namespace

void CLI::run() {
    while (true) {
        clearScreen();
        std::cout << "=== Digit OCR ===\n";
        std::cout << "Current algorithm: " << algorithmName(currentAlgorithm) << "\n\n";
        std::cout << "1. Train KNN | Neural Network Model\n";
        std::cout << "2. Test on MNIST\n";
        std::cout << "3. Test on Real Images\n";
        std::cout << "4. Benchmark Performance\n";
        std::cout << "5. Run Algorithm Tests\n";
        std::cout << "6. Exit\n";
        std::cout << "Choose: ";

        unsigned short choice = 0;
        std::cin >> choice;

        switch (choice) {
        case 1:
            trainingMenu();
            break;
        case 2:
            testingMenu();
            break;
        case 3:
            realImageMenu();
            break;
        case 4:
            pressAnyKeyToContinue();
            break;
        case 5:
            benchmarkMenu();
            break;
        case 6:
            std::cout << "Goodbye!\n";
            return;
        default:
            std::cout << "Invalid choice! Please try again.\n";
            pressAnyKeyToContinue();
            break;
        }
    }
}

void CLI::trainingMenu() {
    clearScreen();

    std::cout << "=== Training Menu ===\n";
    std::cout << "1. Train new model\n";
    std::cout << "2. Load existing model\n";
    std::cout << "3. Back to main menu\n";
    std::cout << "Choose: ";

    unsigned short choice = 0;
    std::cin >> choice;

    if (choice == 1) {
        std::cout << "=== Choose model type ===\n";
        std::cout << "1. Neural Network\n";
        std::cout << "2. KNN Classifier\n";

        unsigned short algorithmChoice = 0;
        std::cin >> algorithmChoice;

        currentAlgorithm = (algorithmChoice == 1 ? AlgorithmType::NEURAL_NETWORK : AlgorithmType::KNN);

        std::string dataPath;
        std::cout << "Enter path to MNIST data folder: ";
        std::cin >> dataPath;

        ocr.trainModel(dataPath, currentAlgorithm);
        std::cout << "Training completed for " << algorithmName(currentAlgorithm) << ".\n";
    } else if (choice == 2) {
        std::cout << "=== Choose model type to load ===\n";
        std::cout << "1. Neural Network\n";
        std::cout << "2. KNN Classifier\n";

        unsigned short algorithmChoice = 0;
        std::cin >> algorithmChoice;

        currentAlgorithm = (algorithmChoice == 1 ? AlgorithmType::NEURAL_NETWORK : AlgorithmType::KNN);

        std::string filename;
        std::cout << "Enter model filename (default: trained_model.dat): ";
        std::cin.ignore();
        std::getline(std::cin, filename);

        if (filename.empty()) {
            filename = (currentAlgorithm == AlgorithmType::NEURAL_NETWORK)
                ? "digit_model.bin"
                : "trained_model.dat";
        }

        ocr.loadModel(filename, currentAlgorithm);
        std::cout << "Model loaded successfully for " << algorithmName(currentAlgorithm) << "!\n";
    }
}

void CLI::testingMenu() {
    clearScreen();
    std::cout << "=== Testing on MNIST ===\n";
    std::cout << "Selected algorithm: " << algorithmName(currentAlgorithm) << "\n";

    if (!ocr.isTrained(currentAlgorithm)) {
        std::cerr << "Error: Selected model is not trained yet! Please train or load it first.\n";
        pressAnyKeyToContinue();
        return;
    }

    std::cout << "1. Test on training data (self-consistency)\n";
    std::cout << "2. Test on test data (true performance)\n";
    std::cout << "3. Back to main menu\n";
    std::cout << "Choose: ";

    unsigned short choice = 0;
    std::cin >> choice;

    if (choice == 1) {
        std::cout << "Training-data evalutaion is currently implemented only for KNN.\n";
    } else if (choice == 2) {
        std::string dataPath;
        std::cout << "Enter path to MNIST data folder: ";
        std::cin >> dataPath;

        std::cout << "Evaluating on test data...\n";
        const float accuracy = ocr.evaluateOnTestData(dataPath, currentAlgorithm);

        std::cout << "\n=== Performance Analysis ===\n";
        if (accuracy > 0.95f) {
            std::cout << "Excellent performance!\n";
        } else if (accuracy > 0.90f) {
            std::cout << "Good performance!\n";
        } else if (accuracy > 0.85f) {
            std::cout << "Average performance - consider tuning parameters\n";
        } else {
            std::cout << "Poor performance - check feature extraction, preprocessing, or hyperparameters\n";
        }
    }

    pressAnyKeyToContinue();
}

void CLI::clearScreen() {}

void CLI::showMainMenu() {}

void CLI::realImageMenu() {
    clearScreen();
    std::cout << "=== Real Image Recognition ===\n";
    std::cout << "Selected algorithm: " << algorithmName(currentAlgorithm) << "\n";
    std::cout << "Enter BMP file path: ";

    std::string path;
    std::cin >> path;

    ImageMatrix img;
    if (!BMPReader::loadBMP(path, img)) {
        std::cerr << "Failed to load image!\n";
        pressAnyKeyToContinue();
        return;
    }

    if (!ocr.isTrained(currentAlgorithm)) {
        std::cerr << "Selected model is not trained or loaded yet!\n";
        pressAnyKeyToContinue();
        return;
    }

    std::cout << "Recognizing digits...\n";
    const std::string prediction = ocr.recognize(img, currentAlgorithm);
    std::cout << "Detected number: " << prediction << "\n";
    pressAnyKeyToContinue();
}

void CLI::benchmarkMenu() {
    clearScreen();
    std::cout << "=== Testing Menu ===\n";
    std::cout << "1. Test KNN Algorithm\n";

    unsigned short choice = 0;
    std::cin >> choice;

    if (choice == 1) {
        testSuite.testKNN();
    }
}

void CLI::pressAnyKeyToContinue() {}
