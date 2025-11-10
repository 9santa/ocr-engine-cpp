cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(DigitOCR)

set(CMAKE_CXX_STANDARD 17)


include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_FILES "src/*.cpp" "src/*/*.cpp")

# remove two unfinished files
# list(REMOVE_ITEM SRC_FILES
#     "${CMAKE_SOURCE_DIR}/src/neural_network.cpp"
#     "${CMAKE_SOURCE_DIR}/src/gradient.cpp"
# )

add_executable(ocr_engine ${SRC_FILES})

target_include_directories(ocr_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

