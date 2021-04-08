__device__ void imageIntegral(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height);

__device__ void verticalSum(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height);

__device__ void horizontalSum(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height);

__global__ void bradleyBinarization(unsigned char *inputArray, unsigned int *sumMat, unsigned char *outputArray, unsigned int width,
                                    unsigned int height) {
    unsigned int S, s2, T; //todo add initialization
    imageIntegral(inputArray, sumMat, width, height);
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < height * width; i += blockDim.x) { //todo remove after finish
        outputArray[i] = sumMat[i] / (256 * 16);
    }
    //todo parallel loop
    for (unsigned int col = 0; col < width; col++) {
        for (unsigned int row = 0; row < height; row++) {
            unsigned int y0 = max(row-s2,0); //todo format checks
            unsigned int y1 = min(row+s2, height-1);
            unsigned int x0 = max(col-s2,0);
            unsigned int x1 = min(col+s2, width-1);
            unsigned int count = (y1-y0)*(x1-x0); //todo finish alghoritm
//            unsigned int sum = ....;
//                if
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
            unsigned int index = width * row + col;
            outputArray[index] = inputArray[index] + outputArray[index - width];
        }
    }
}

__device__ void horizontalSum(const unsigned char *inputArray, unsigned int *outputArray, unsigned int width, unsigned int height) {
    for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; row < height; row += blockDim.x) {
        for (unsigned int col = 1; col < width; col++) {
            unsigned int index = width * row + col;
            outputArray[index] = inputArray[index] + outputArray[index - 1];
        }
    }
}
