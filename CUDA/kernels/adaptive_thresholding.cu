#include "adaptive_thresholding.cuh"

__constant__ int width;
__constant__ int height;
__constant__ int S2;
__constant__ double T;

__device__ int getIndex(int col, int row) {
    return width * row + col;
}

__device__ int maxInt(const int a, const int b) {
    return a > b ? a : b;
}

__device__ int minInt(const int a, const int b) {
    return a < b ? a : b;
}

__global__ void verticalSum(const unsigned char *inputArray, unsigned int *outputArray) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        outputArray[col] = inputArray[col];
        for (int row = 1; row < height; row++) {
            int index = getIndex(col, row);
            outputArray[index] = inputArray[index] + outputArray[index - width];
        }
    }
}

__global__ void horizontalSum(unsigned int *outputArray) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height) {
        for (int col = 1; col < width; col++) {
            int index = getIndex(col, row);
            outputArray[index] = outputArray[index] + outputArray[index - 1];
        }
    }
}

__global__ void binarization(const unsigned char *inputArray, const unsigned int *sumMat, unsigned char *outputArray) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) {
        const int col = index % width;
        const int row = index / width;

        int x0 = maxInt(col - S2, 0);
        int x1 = minInt(col + S2, width - 1);
        int y0 = maxInt(row - S2, 0);
        int y1 = minInt(row + S2, height - 1);

        int count = (x1 - x0) * (y1 - y0);
        long sum = sumMat[getIndex(x1, y1)] - sumMat[getIndex(x1, y0)] - sumMat[getIndex(x0, y1)] + sumMat[getIndex(x0, y0)];

        if (inputArray[getIndex(col, row)] * count < (int) (sum * (1.0 - T)))
            outputArray[getIndex(col, row)] = 0;
        else
            outputArray[getIndex(col, row)] = 255;
    }
}