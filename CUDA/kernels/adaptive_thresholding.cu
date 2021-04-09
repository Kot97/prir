#include "adaptive_thresholding.cuh"

__constant__ int width;
__constant__ int height;
__constant__ unsigned int s2;
__constant__ unsigned int T;

__device__ void imageIntegral(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height);

__device__ void verticalSum(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height);

__device__ void horizontalSum(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height);

__device__ unsigned int getIndex(unsigned int col, unsigned int row);

__global__ void bradleyBinarization(unsigned char *inputArray, unsigned int *sumMat, unsigned char *outputArray) {
    imageIntegral(inputArray, sumMat, width, height);

    //todo full parallel loop
    //todo fix loop logic
    for (unsigned int col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += blockDim.x) {
        for (unsigned int row = 0; row < height; row++) {
            unsigned int y0 = max(row - s2, 0); //todo format checks
            unsigned int y1 = min(row + s2, height - 1);
            unsigned int x0 = max(col - s2, 0);
            unsigned int x1 = min(col + s2, width - 1);
            unsigned int count = (y1 - y0) * (x1 - x0);
            unsigned int sum =
                    sumMat[getIndex(x1, y1)] - sumMat[getIndex(x1, y0)] - sumMat[getIndex(x0, y1)] + sumMat[getIndex(x0, y0)];
            if (inputArray[getIndex(col, row)] * count < sum * (100. - T) / 100.)
                outputArray[getIndex(col, row)] = 0;
            else
                outputArray[getIndex(col, row)] = 255;
        }
    }
}

__device__ void imageIntegral(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height) {
    verticalSum(inputArray, outputArray, width, height);
    horizontalSum(inputArray, outputArray, width, height);
}

__device__ void verticalSum(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height) {
    for (unsigned int col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += blockDim.x) {
        outputArray[col] = inputArray[col];
        for (unsigned int row = 1; row < height; row++) {
            unsigned int index = getIndex(col, row);
            outputArray[index] = inputArray[index] + outputArray[index - width];
        }
    }
}

__device__ void horizontalSum(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height) {
    for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; row < height; row += blockDim.x) {
        for (unsigned int col = 1; col < width; col++) {
            unsigned int index = getIndex(col, row);
            outputArray[index] = inputArray[index] + outputArray[index - 1];
        }
    }
}

__device__ unsigned int getIndex(unsigned int col, unsigned int row) {
    return width * row + col;
}