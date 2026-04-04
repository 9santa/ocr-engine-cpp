#pragma once
#ifndef CLI_H
#define CLI_H

#include "app/digit_ocr.h"
#include "../../test/test_suite.h"

class CLI {
public:
    void run();

private:
    void showMainMenu();
    void trainingMenu();
    void testingMenu();
    void realImageMenu();
    void benchmarkMenu();

    void clearScreen();
    void pressAnyKeyToContinue();

    DigitOCR ocr;
    TestSuite testSuite;
    AlgorithmType currentAlgorithm = AlgorithmType::KNN;
};


#endif // !CLI_H
