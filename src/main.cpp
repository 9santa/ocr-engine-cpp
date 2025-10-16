#include "digit_ocr.h"
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

signed main(void) {
    DigitOCR ocr;



    cv::Mat image = cv::imread("test_digits.png");
    if (image.empty()) {
        std::cout << "Could not load test image!\n";
        return -1;
    }

    // Recognize digits
    std::string result = ocr.recognize(image);
    std::cout << "Recognized digits: " << result << "\n";

    // Display result
    cv::putText(image, "Digits: " + result, cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX,
            1, cv::Scalar(0, 255, 0), 2);

    cv::imshow("OCR Result", image);
    cv::waitKey(0);

    return 0;
}
