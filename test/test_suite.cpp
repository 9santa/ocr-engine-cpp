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
        {{0.1f, 0.1f}, 0},
        {{0.2f, 0.2f}, 0},
        {{0.8f, 0.8f}, 1},
        {{0.9f, 0.9f}, 1}
    };
    knn1.train(&data1);

    // Test predictions
    assertTrue(knn1.predict({0.15f, 0.15f}) == 0, "KNN Simple Case - Class 0");
    assertTrue(knn1.predict({0.85f, 0.85f}) == 1, "KNN Simple Case - Class 1");

    // Test accuracy
    float accuracy1 = knn1.evaluate(data1);
    assertTrue(accuracy1 > 0.99f, "KNN Accuracy");

    std::cout << "Basic 2-class test passed\n";


    // Test Case 2: 3-class problem
    KNNClassifier knn2(3);
    std::vector<TrainingSample> data2 = {
        {{0.0f, 0.0f}, 0},
        {{0.1f, 0.1f}, 0},

        {{1.0f, 1.0f}, 1},
        {{1.1f, 1.1f}, 1},

        {{0.0f, 1.0f}, 2},
        {{0.1f, 1.1f}, 2}
    };
    knn2.train(&data2);

    // Test predictions for each class
    assertTrue(knn2.predict({0.05f, 0.05f}) == 0, "KNN 3-class - Class 0");
    assertTrue(knn2.predict({1.05f, 1.05f}) == 1, "KNN 3-class - Class 1");
    assertTrue(knn2.predict({0.05f, 1.05f}) == 2, "KNN 3-class - Class 2");    std::cout << "3-class test passed\n";

    float accuracy2 = knn2.evaluate(data2);
    assertTrue(accuracy2 > 0.99f, "KNN 3-class accuracy on train set");

    sleep(1);
    std::cout << "\nKNN test finished\n\n";
}
