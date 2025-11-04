#include "../include/cli.h"
#include "../include/digit_ocr.h"
#include "../include/bmp_reader.h"
#include <iostream>


void CLI::run() {
    while(true) {
        clearScreen();
        std::cout << "=== Digit OCR ===\n";
        std::cout << "1. Train Model\n";
        std::cout << "2. Test on MNIST\n"; 
        std::cout << "3. Test on Real Images\n";
        std::cout << "4. Run Algorithm Tests\n";
        std::cout << "5. Benchmark Performance\n";
        std::cout << "6. Exit\n";
        std::cout << "Choose: ";

        short choice; std::cin >> choice;

        switch (choice) {
            case 1: trainingMenu(); break;
            case 2: testingMenu(); break;
            case 3: realImageMenu(); break;
            case 4:
                testSuite.runAllTests();
                pressAnyKeyToContinue();
                break;
            case 5: benchmarkMenu(); break;
            case 6: return;
            default:
                std::cout << "Invalid choice!\n";
                pressAnyKeyToContinue();
        }
    }
}
