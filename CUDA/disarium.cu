#include <iostream>
#include <builtin_types.h>
#include "kernels/disarium_number.cuh"

const unsigned long NUMBERS_COUNT = 1000000000;

unsigned long *allocateArrayOnGPU(unsigned long elementsCount, size_t elementSize);

void synchronizeKernel();

int main(int argc, char **argv) {
    unsigned long *generatedNumbersGPU = allocateArrayOnGPU(NUMBERS_COUNT, sizeof(unsigned long));
    unsigned long *generatedNumbersCPU = new unsigned long[NUMBERS_COUNT];

    unsigned long threadNum = 1024;
    unsigned long blocksNum = 1;
    while (NUMBERS_COUNT/threadNum > blocksNum)
        blocksNum*=4;
    std::cout<< "block num: " << blocksNum << std::endl << "threads count: " << threadNum << std::endl;
    generateDisariumNumbers <<< blocksNum, threadNum >>>(generatedNumbersGPU, NUMBERS_COUNT);

    synchronizeKernel();

    cudaMemcpy(generatedNumbersCPU, generatedNumbersGPU, sizeof(unsigned long) * NUMBERS_COUNT, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < NUMBERS_COUNT; i++)
        if (generatedNumbersCPU[i] == 1)
            std::cout << i << std::endl;

    cudaFree(generatedNumbersGPU);
    delete[] generatedNumbersCPU;

    return 0;
}

void synchronizeKernel() {
    cudaError_t errorCode;
    if ((errorCode = cudaDeviceSynchronize()) != cudaSuccess) {
        std::cout << "error during Device Synchronize: " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

unsigned long *allocateArrayOnGPU(const unsigned long elementsCount, const size_t elementSize) {
    cudaError_t errorCode;
    unsigned long *table_addr;
    if ((errorCode = cudaMalloc((void **) &table_addr, elementsCount * elementSize)) != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    return table_addr;
}


