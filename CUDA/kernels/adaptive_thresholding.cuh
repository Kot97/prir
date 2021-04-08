#ifndef PRIR_ADAPTIVE_THRESHOLDING_CUH
#define PRIR_ADAPTIVE_THRESHOLDING_CUH

__global__ void bradleyBinarization(unsigned char *inputArray, unsigned int *sumMat, unsigned char *outputArray,
                                    unsigned int width, unsigned int height);

#endif //PRIR_ADAPTIVE_THRESHOLDING_CUH
