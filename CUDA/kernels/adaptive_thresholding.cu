#include "nppi_statistics_functions.h"

__global__ void bradleyBinarization(unsigned char *inputArray, unsigned char *bufferArray, unsigned char *outputArray,
                                    unsigned int width, unsigned int height) {
//    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < height * width; i += blockDim.x * gridDim.x) {
//        outputArray[i] = (i % width) / 16;
//    }

    NppiSize oROI;

    nppiIntegral_8u32f_C1R(inputArray, width, bufferArray, width + 1, oROI, 0);

//    cv::Mat sumMat, result = cv::Mat::zeros(image.size(), CV_8UC1);
//
//    int s = max(height, width) / 16;
//    double T = 0.15;
//
//    unsigned char *p_inputMat, *p_outputMat;
//    int x1, y1, x2, y2, count, sum;
//    int *p_y1, *p_y2;


//    integral(inputArray, bufferArray, width, height);
//
//    cv::integral(image, sumMat);

//    for (int i = 0; i < height; ++i){
//        y1 = i - s;
//        y2 = i + s;
//
//        if (y1 < 0) y1 = 0;
//        if (y2 >= height) y2 = height - 1;

//        p_y1 = sumMat.ptr<int>(y1);
//        p_y2 = sumMat.ptr<int>(y2);
//        p_inputMat = image.ptr<uchar>(i);
//        p_outputMat = result.ptr<uchar>(i);

//        for (int j = 0; j < width; ++j){
//            x1 = j - s;
//            x2 = j + s;
//
//            if (x1 < 0) x1 = 0;
//            if (x2 >= width) x2 = width - 1;
//
//            count = (x2 - x1) * (y2 - y1);
//
//            // I(x,y)= s(x2,y2) - s(x1,y2) - s(x2,y1) + s(x1,x1)
//            sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];
//
//            if (static_cast<int>(p_inputMat[j] * count) < static_cast<int>(sum * (1.0 - T)))
//                p_outputMat[j] = 255;
//            else
//                p_outputMat[j] = 0;
//        }
//    }
}
