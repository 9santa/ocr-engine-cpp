cmake_minimum_required(VERSION 3.10)
project(DigitOCR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(ocr_engine
    src/app/main.cpp
    src/app/cli.cpp
    src/app/digit_ocr.cpp

    src/core/image_matrix.cpp
    src/data/mnist_loader.cpp
    src/io/bmp_reader.cpp
    src/baselines/knn/feature_extractor.cpp
    src/baselines/knn/knn_classifier.cpp
    src/baselines/neural_network/neural_network_classifier.cpp
    src/baselines/neural_network/nn/loss.cpp
    src/baselines/neural_network/nn/opt_layer.cpp
    src/baselines/neural_network/nn/opt_mlp.cpp
    src/baselines/neural_network/nn/opt_neuron.cpp
    src/preprocess/preprocessor.cpp
    test/test_suite.cpp
)


target_include_directories(ocr_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
