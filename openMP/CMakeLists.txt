cmake_minimum_required(VERSION 3.10)
project(prir_openmp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bradley bradley.cpp lib/adaptive_thresholding.cpp)
target_link_libraries(bradley PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(disarium disarium.cpp lib/disarium_number.cpp)
target_link_libraries(disarium PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
