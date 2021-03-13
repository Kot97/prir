cmake_minimum_required(VERSION 3.10)
project(PRiR)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE
        lib/adaptive_thresholding.cpp)

set(HEADERS
        lib/benchmarking.hpp
        lib/adaptive_thresholding.hpp)

add_executable(bradley bradley.cpp ${SOURCE} ${HEADERS})
target_link_libraries(bradley PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)