__device__ void
imageIntegral(unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height) {
    for (unsigned int col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += blockDim.x) {
        outputArray[col] = inputArray[col];
        for (unsigned int row = 1; row < height; row++) {
            unsigned int index = width * row + col;
            outputArray[index] = inputArray[index] + outputArray[index - width];
        }
    }
    for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; row < height; row += blockDim.x) {
        for (unsigned int col = 1; col < width; col++) {
            unsigned int index = width * row + col;
            outputArray[index] = inputArray[index] + outputArray[index - 1];
        }
    }
}

__global__ void bradleyBinarization(unsigned char *inputArray, unsigned int *sumMat, unsigned char *outputArray,

                                    unsigned int width, unsigned int height) {

    imageIntegral(inputArray, sumMat, width, height);
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < height * width; i += blockDim.x) {
        outputArray[i] = sumMat[i] / (256 * 16);
    }

    for (unsigned int col = 0; col < width; col++) {
        for (unsigned int row = 0; row < height; row++) {
            unsigned int index = row * width + col;
        }
    }
}
