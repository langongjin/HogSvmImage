cmake_minimum_required(VERSION 3.6)
project(ColorHogSVM)

FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(ColorHogSVM ${SOURCE_FILES})

target_link_libraries(ColorHogSVM ${OpenCV_LIBS})