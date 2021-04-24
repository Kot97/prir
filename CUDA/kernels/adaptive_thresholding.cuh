#ifndef PRIR_ADAPTIVE_THRESHOLDING_CUH
#define PRIR_ADAPTIVE_THRESHOLDING_CUH

extern __constant__ int width;
extern __constant__ int height;
extern __constant__ int S2;
extern __constant__ double T;

__global__ void verticalSum(const unsigned char *inputArray, unsigned int *outputArray);

__global__ void horizontalSum(unsigned int *outputArray);

__global__ void binarization(const unsigned char *inputArray, const unsigned int *sumMat, unsigned char *outputArray);


#endif //PRIR_ADAPTIVE_THRESHOLDING_CUH
