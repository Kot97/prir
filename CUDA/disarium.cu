#include <iostream>
#include <builtin_types.h>
#include "kernels/disarium_number.cuh"
#include "cudaUtils.cuh"

const unsigned int NUMBERS_COUNT = 12;

void printResult(const unsigned long *generatedNumbersCPU);

int main(int argc, char **argv) {
    unsigned long *generatedNumbersGPU = allocateArrayOnGPU<unsigned long>(NUMBERS_COUNT);
    unsigned long *generatedNumbersCPU = new unsigned long[NUMBERS_COUNT];

    //todo update thread/blocks calculator
    unsigned long threadNum = 1;
    unsigned long blocksNum = 1;
    std::cout << "block num: " << blocksNum << std::endl << "threads count: " << threadNum << std::endl;
    generateDisariumNumbers <<< blocksNum, threadNum >>>(generatedNumbersGPU, NUMBERS_COUNT);

    synchronizeKernel();
    transferDataFromGPU<unsigned long>(generatedNumbersGPU, generatedNumbersCPU, NUMBERS_COUNT);
    printResult(generatedNumbersCPU);

    cudaFree(generatedNumbersGPU);
    delete[] generatedNumbersCPU;

    return 0;
}

void printResult(const unsigned long *generatedNumbersCPU) {
    for (unsigned int i = 0; i < NUMBERS_COUNT; i++)
        std::cout << generatedNumbersCPU[i] << std::endl;
}