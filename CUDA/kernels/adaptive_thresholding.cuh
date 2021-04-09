#ifndef PRIR_ADAPTIVE_THRESHOLDING_CUH
#define PRIR_ADAPTIVE_THRESHOLDING_CUH

extern __constant__ int width;
extern __constant__ int height;
extern __constant__ unsigned int s2;
extern __constant__ unsigned int T;

__global__ void bradleyBinarization(unsigned char *inputArray, unsigned int *sumMat, unsigned char *outputArray);

#endif //PRIR_ADAPTIVE_THRESHOLDING_CUH
