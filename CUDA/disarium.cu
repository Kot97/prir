#include <iostream>
#include <builtin_types.h>
#include "kernels/disarium_number.cuh"
#include "cudaUtils.cuh"

const unsigned int NUMBERS_COUNT = 19;

void printResult(const unsigned long *generatedNumbersCPU);

int main(int argc, char **argv) {
    unsigned long *generatedNumbersGPU = allocateArrayOnGPU<unsigned long>(NUMBERS_COUNT);
    unsigned long *generatedNumbersCPU = new unsigned long[NUMBERS_COUNT];
    //todo update thread/blocks calculator
    unsigned long threadNum = 128;
    unsigned long blocksNum = 256;
    std::cout << "block num: " << blocksNum << std::endl << "threads count: " << threadNum << std::endl;
    generateDisariumNumbers <<< blocksNum, threadNum >>>(generatedNumbersGPU, NUMBERS_COUNT);

    synchronizeKernel();
    transferDataFromGPU<unsigned long>(generatedNumbersCPU, generatedNumbersGPU, NUMBERS_COUNT);
    printResult(generatedNumbersCPU);

    cudaFree(generatedNumbersGPU);
    delete[] generatedNumbersCPU;

    return 0;
}

void printResult(const unsigned long *generatedNumbersCPU) {
    for (unsigned int i = 0; i < NUMBERS_COUNT; i++)
        std::cout << generatedNumbersCPU[i] << std::endl;
}