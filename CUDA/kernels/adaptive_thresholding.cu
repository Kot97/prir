#include <cuda_runtime.h>
#include "adaptive_thresholding.cuh"

__constant__ int width;
__constant__ int height;
__constant__ int S2;
__constant__ double T;

__device__ void imageIntegral(const unsigned char *inputArray, unsigned int *outputArray);

__device__ void verticalSum(const unsigned char *inputArray, unsigned int *outputArray);

__device__ void horizontalSum(const unsigned char *inputArray, unsigned int *outputArray);

__device__ void binarization(const unsigned char *inputArray, const unsigned int *sumMat, unsigned char *outputArray);

__device__ unsigned int getIndex(int col, int row) {
    return width * row + col;
}

__device__ int maxInt(int a, int b) {
    return a > b ? a : b;
}

__device__ int minInt(int a, int b) {
    return a < b ? a : b;
}

__global__ void bradleyBinarization(unsigned char *inputArray, unsigned int *sumMat, unsigned char *outputArray) {
    imageIntegral(inputArray, sumMat);
    binarization(inputArray, sumMat, outputArray);
}

__device__ void imageIntegral(const unsigned char *inputArray, unsigned int *outputArray) {
    verticalSum(inputArray, outputArray);
    horizontalSum(inputArray, outputArray);
}

__device__ void verticalSum(const unsigned char *inputArray, unsigned int *outputArray) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += blockDim.x) {
        outputArray[col] = inputArray[col];
        for (int row = 1; row < height; row++) {
            unsigned int index = getIndex(col, row);
            outputArray[index] = inputArray[index] + outputArray[index - width];
        }
    }
}

__device__ void horizontalSum(const unsigned char *inputArray, unsigned int *outputArray) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < height; row += blockDim.x) {
        for (int col = 1; col < width; col++) {
            unsigned int index = getIndex(col, row);
            outputArray[index] = inputArray[index] + outputArray[index - 1];
        }
    }
}

__device__ void binarization(const unsigned char *inputArray, const unsigned int *sumMat, unsigned char *outputArray) {
    //todo full parallel loop
    //todo fix loop logic
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += blockDim.x) {
        int x0 = maxInt(col - S2, 0);
        int x1 = minInt(col + S2, width - 1);
        for (int row = 0; row < height; row++) {
            int y0 = maxInt(row - S2, 0);
            int y1 = minInt(row + S2, height - 1);
            int count = (x1 - x0)*(y1 - y0);
            unsigned int sum =
                    sumMat[getIndex(x1, y1)] - sumMat[getIndex(x1, y0)] - sumMat[getIndex(x0, y1)] +
                    sumMat[getIndex(x0, y0)]; //todo is it bug??
            if (inputArray[getIndex(col, row)] * count < sum * (100. - T) / 100.)
                outputArray[getIndex(col, row)] = 0;
            else
                outputArray[getIndex(col, row)] = 255;
        }
    }
}