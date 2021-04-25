#include <iostream>
#include <builtin_types.h>
#include <stdlib.h>
#include "kernels/disarium_number.cuh"
#include "cudaUtils.cuh"

const unsigned int NUMBERS_COUNT = 100000;

const unsigned int THREADS_NUM = 128;

void runKernel(unsigned int *generatedNumbersGPU, bool *resultGPU);

void generateNumbers(unsigned int *table, unsigned int n);

void printResult(const unsigned int *generatedNumbers, const bool *result);

int main(int argc, char **argv) {
    unsigned int *generatedNumbersCPU = new unsigned int[NUMBERS_COUNT];
    generateNumbers(generatedNumbersCPU, NUMBERS_COUNT);

    unsigned int *generatedNumbersGPU = allocateArrayOnGPU<unsigned int>(NUMBERS_COUNT);
    transferDataToGPU(generatedNumbersGPU, generatedNumbersCPU, NUMBERS_COUNT);

    bool *resultGPU = allocateArrayOnGPU<bool>(NUMBERS_COUNT);

    runKernel(generatedNumbersGPU, resultGPU);

    bool *resultCPU = new bool[NUMBERS_COUNT];
    transferDataFromGPU<bool>(resultCPU, resultGPU, NUMBERS_COUNT);

    freeArrayGPU(resultGPU);
    freeArrayGPU(generatedNumbersGPU);

    printResult(generatedNumbersCPU, resultCPU);
    delete[] generatedNumbersCPU;
    delete[] resultCPU;

    return 0;
}

void generateNumbers(unsigned int *table, unsigned int n) {
    srand(time(NULL));
    for (unsigned int i = 0; i < n; i++)
        table[i] = rand();
}

void runKernel(unsigned int *generatedNumbersGPU, bool *resultGPU) {
    unsigned int blocksNumber = getBlocksNumber(THREADS_NUM, NUMBERS_COUNT);
    std::cout << "block num: " << blocksNumber << std::endl << "threads count: " << THREADS_NUM << std::endl;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start, 0);

    generateDisariumNumbers <<< blocksNumber, THREADS_NUM >>>(generatedNumbersGPU, resultGPU, NUMBERS_COUNT);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate: %f ms \n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void printResult(const unsigned int *generatedNumbers, const bool *result) {
    std::cout << "Results: " << std::endl;
    for (unsigned int i = 0; i < NUMBERS_COUNT; i++)
        if (result[i])
            std::cout << generatedNumbers[i] << std::endl;
    std::cout << "----------" << std::endl;
}
