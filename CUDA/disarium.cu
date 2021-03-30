#include <iostream>
#include <builtin_types.h>
#include <stdlib.h>
#include "kernels/disarium_number.cuh"
#include "cudaUtils.cuh"

const unsigned int NUMBERS_COUNT = 100000;

const unsigned int THREADS_NUM = 128;

unsigned int getBlocksNumber(const unsigned int threadsNum, const unsigned int numbersCount) {
    return ceil(numbersCount / threadsNum) + 1;
}

void printResult(const unsigned int *generatedNumbers, const bool *result);

void generateNumbers(unsigned int *table, unsigned int n);

int main(int argc, char **argv) {
    unsigned int *generatedNumbersCPU = new unsigned int[NUMBERS_COUNT];
    generateNumbers(generatedNumbersCPU, NUMBERS_COUNT);

    unsigned int *generatedNumbersGPU = allocateArrayOnGPU<unsigned int>(NUMBERS_COUNT);
    transferDataToGPU(generatedNumbersGPU, generatedNumbersCPU, NUMBERS_COUNT);

    bool *resultGPU = allocateArrayOnGPU<bool>(NUMBERS_COUNT);

    unsigned int blocksNumber = getBlocksNumber(THREADS_NUM, NUMBERS_COUNT);
    std::cout << "block num: " << blocksNumber << std::endl << "threads count: " << THREADS_NUM << std::endl;

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    generateDisariumNumbers <<< blocksNumber, THREADS_NUM >>>(generatedNumbersGPU, resultGPU, NUMBERS_COUNT);

//    synchronizeKernel(); //todo check is synch work
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.5f ms \n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    bool *resultCPU = new bool[NUMBERS_COUNT];
    transferDataFromGPU<bool>(resultCPU, resultGPU, NUMBERS_COUNT);

    freeArrayGPU(resultGPU);
    freeArrayGPU(generatedNumbersGPU);

    printResult(generatedNumbersCPU, resultCPU);
    delete[] generatedNumbersCPU;
    delete[] resultCPU;

    return 0;
}

void printResult(const unsigned int *generatedNumbers, const bool *result) {
    std::cout << "Results: " << std::endl;
    for (unsigned int i = 0; i < NUMBERS_COUNT; i++)
        if (result[i])
            std::cout << generatedNumbers[i] << std::endl;
    std::cout << "----------" << std::endl;
}

void generateNumbers(unsigned int *table, unsigned int n) {
    srand(time(NULL));
    for (unsigned int i = 0; i < n; i++)
        table[i] = rand();
}
