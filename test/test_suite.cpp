#include "test_suite.h"
#include "../include/baselines/knn/knn_classifier.h"
#include <iostream>
#include <unistd.h>
#include <cmath>

void TestSuite::assertTrue(bool condition, const std::string& testName) {
    if (condition) {
        std::cout << "[PASS] " << testName << "\n";
    } else {
        std::cerr << "[FAIL] " << testName << "\n";
    }
}

void TestSuite::assertEquals(float actual, float expected, float tolerance, const std::string& testName) {
    if (std::fabs(actual - expected) <= tolerance) {
        std::cout << "[PASS] " << testName 
                  << " (Expected: " << expected << ", Actual: " << actual << ")\n";
    } else {
        std::cerr << "[FAIL] " << testName 
                  << " (Expected: " << expected << ", Actual: " << actual << ")\n";
    }
}

void TestSuite::testKNN() {
    std::cout << "\n=== Test: KNN CLassifier ===\n";

    // Test Case 1: Simple 2-class problem
    KNNClassifier knn1(3);
    std::vector<TrainingSample> data1 = {
        {{0.1, 0.1}, 0},
        {{0.2, 0.2}, 0},
        {{0.8, 0.8}, 1},
        {{0.9, 0.9}, 1}
    };
    knn1.train(&data1);

    // Test predictions
    assertTrue(knn1.predict({0.15, 0.15}) == 0, "KNN Simple Case - Class 0");
    assertTrue(knn1.predict({0.85, 0.85}) == 1, "KNN Simple Case - Class 1");

    // Test accuracy
    float accuracy1 = knn1.evaluate(data1);
    assertTrue(accuracy1 > 0.9f, "KNN Accuracy");

    std::cout << "Basic 2-class test passed\n";


    // Test Case 2: 3-class problem
    KNNClassifier knn2(3);
    std::vector<TrainingSample> data2 = {
        {{0.0, 0.0}, 1},
        {{0.25, 0.25}, 1},
        {{0.0, 0.5}, 1},
        {{0.5, 0.5}, 1},
        {{0.0, 0.5}, 2},
        {{0.0, 1.0}, 2},
    };
    knn2.train(&data2);

    // Test predictions for each class
    assertTrue(knn2.predict({0.1, 0.1}) == 1, "KNN 3-class - Class 1");
    assertTrue(knn2.predict({0.5, 0.2}) == 2, "KNN 3-class - Class 2");
    assertTrue(knn2.predict({0.0, 0.7}) == 3, "KNN 3-class - Class 3");

    std::cout << "3-class test passed\n";

    sleep(1);
    std::cout << "\nKNN test finished\n\n";
}
