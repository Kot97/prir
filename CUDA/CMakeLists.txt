cmake_minimum_required(VERSION 3.10)
#fix test server config
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(PRiR LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bradley bradley.cu cudaUtils.cuh)
target_link_libraries(bradley PRIVATE ${OpenCV_LIBS})
set_target_properties(
        bradley
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(disarium disarium.cu kernels/disarium_number.cu kernels/disarium_number.cuh cudaUtils.cuh)
set_target_properties(
        disarium
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if ( CMAKE_COMPILER_IS_GNUCC )
    target_compile_options(disarium PRIVATE "-Wall -Wextra")
endif()