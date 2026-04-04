#include "app/cli.h"

#include "io/bmp_reader.h"

#include <iostream>
#include <string>

void CLI::run() {
    while (true) {
        clearScreen();
        std::cout << "=== Digit OCR ===\n";
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
        const AlgorithmType type =
            (algorithmChoice == 1 ? AlgorithmType::NEURAL_NETWORK : AlgorithmType::KNN);

        std::string dataPath;
        std::cout << "Enter path to MNIST data folder: ";
        std::cin >> dataPath;

        ocr.trainModel(dataPath, type);
        std::cout << "Training completed.\n";
    } else if (choice == 2) {
        std::string filename;
        std::cout << "Enter model filename (default: trained_model.dat): ";
        std::cin.ignore();
        std::getline(std::cin, filename);

        if (filename.empty()) {
            filename = "trained_model.dat";
        }

        ocr.loadModel(filename);
        std::cout << "Model loaded successfully!\n";
    }
}

void CLI::testingMenu() {
    clearScreen();
    std::cout << "=== Testing on MNIST ===\n";

    if (!ocr.isTrained()) {
        std::cerr << "Error: Model not trained yet! Please train first!\n";
        pressAnyKeyToContinue();
        return;
    }

    std::cout << "1. Test on training data (self-consistency)\n";
    std::cout << "2. Test on test data (true performance)\n";
    std::cout << "3. Back to main menu\n";
    std::cout << "Choose: ";

    unsigned short choice = 0;
    std::cin >> choice;

    if (choice == 2) {
        std::string dataPath;
        std::cout << "Enter path to MNIST data folder: ";
        std::cin >> dataPath;

        std::cout << "Evaluating on test data...\n";
        const float accuracy = ocr.evaluateOnTestData(dataPath, AlgorithmType::NEURAL_NETWORK);

        std::cout << "\n=== Performance Analysis ===\n";
        if (accuracy > 0.95f) {
            std::cout << "Excellent performance!\n";
        } else if (accuracy > 0.90f) {
            std::cout << "Good performance!\n";
        } else if (accuracy > 0.85f) {
            std::cout << "Average performance - consider tuning parameters\n";
        } else {
            std::cout << "Poor performance - check feature extraction or K value\n";
        }
    }

    pressAnyKeyToContinue();
}

void CLI::clearScreen() {}

void CLI::showMainMenu() {}

void CLI::realImageMenu() {
    clearScreen();
    std::cout << "Enter BMP file path: ";
    std::string path;
    std::cin >> path;

    ImageMatrix img;
    if (!BMPReader::loadBMP(path, img)) {
        std::cerr << "Failed to load image!\n";
        pressAnyKeyToContinue();
        return;
    }

    std::cout << "Recognizing digits...\n";
    const std::string prediction = ocr.recognize(img);
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
