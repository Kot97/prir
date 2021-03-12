cmake_minimum_required(VERSION 3.17)
project(PRiR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(PRiR main.cpp)
target_link_libraries(PRiR ${OpenCV_LIBS})