#include <iostream>
#include <builtin_types.h>
#include "kernels/disarium_number.cuh"

const unsigned int NUMBERS_COUNT = 12;

unsigned long *allocateArrayOnGPU(unsigned long elementsCount, size_t elementSize);

void synchronizeKernel();

void transferDataFromGPU(unsigned long *generatedNumbersGPU, unsigned long *generatedNumbersCPU);

void printResult(const unsigned long *generatedNumbersCPU);

int main(int argc, char **argv) {
    unsigned long *generatedNumbersGPU = allocateArrayOnGPU(NUMBERS_COUNT, sizeof(unsigned long));
    unsigned long *generatedNumbersCPU = new unsigned long[NUMBERS_COUNT];

    //todo update thread/blocks calculator
    unsigned long threadNum = 1;
    unsigned long blocksNum = 1;
    std::cout << "block num: " << blocksNum << std::endl << "threads count: " << threadNum << std::endl;
    generateDisariumNumbers <<< blocksNum, threadNum >>>(generatedNumbersGPU, NUMBERS_COUNT);

    synchronizeKernel();
    transferDataFromGPU(generatedNumbersGPU, generatedNumbersCPU);
    printResult(generatedNumbersCPU);

    cudaFree(generatedNumbersGPU);
    delete[] generatedNumbersCPU;

    return 0;
}

void printResult(const unsigned long *generatedNumbersCPU) {
    for (unsigned int i = 0; i < NUMBERS_COUNT; i++)
        std::cout << generatedNumbersCPU[i] << std::endl;
}

void transferDataFromGPU(unsigned long *generatedNumbersGPU, unsigned long *generatedNumbersCPU) {
    cudaError_t errorCode = cudaMemcpy(generatedNumbersCPU, generatedNumbersGPU, sizeof(unsigned long) * NUMBERS_COUNT,
                                       cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess) {
        std::cout << "error during transfer data from gpu " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

void synchronizeKernel() {
    cudaError_t errorCode = cudaDeviceSynchronize();
    if (errorCode != cudaSuccess) {
        std::cout << "error during Device Synchronize: " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

unsigned long *allocateArrayOnGPU(const unsigned long elementsCount, const size_t elementSize) {
    unsigned long *table_addr;
    cudaError_t errorCode = cudaMalloc((void **) &table_addr, elementsCount * elementSize);
    if (errorCode != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    return table_addr;
}