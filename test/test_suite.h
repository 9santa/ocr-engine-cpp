#pragma once
#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include <string>


class TestSuite {
public:
    TestSuite() = default;
    // void runAllTests();

    // Core algorithms tests
    void testKNN();
    void testEuclideanDistance();
    void testPrepocessingPipeline();

    // Accuracy tests
    void testMNISTAccuracy();

private:
    void assertTrue(bool condition, const std::string& testName);
    void assertEquals(float actual, float expected, float tolerance, const std::string& testName);
};


#endif // !TEST_SUITE_H
