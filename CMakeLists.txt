cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(DigitOCR)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_FILES "src/*.cpp" "src/*/*.cpp")

add_executable(ocr_engine ${SRC_FILES})


target_include_directories(ocr_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ocr_engine PRIVATE ${OpenCV_LIBS})

