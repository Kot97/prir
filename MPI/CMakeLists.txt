cmake_minimum_required(VERSION 3.10)
project(prir_mpi)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})

add_executable(bradley bradley.cpp lib/adaptive_thresholding.cpp)
target_link_libraries(bradley PRIVATE ${OpenCV_LIBS} ${MPI_LIBRARIES})

add_executable(disarium disarium.cpp lib/disarium_number.cpp)
target_link_libraries(disarium PRIVATE ${MPI_LIBRARIES})

add_executable(disarium_serial disarium_serial.cpp lib/disarium_number.cpp)
target_link_libraries(disarium_serial PRIVATE ${MPI_LIBRARIES})
