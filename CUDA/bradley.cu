#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "cudaUtils.cuh"
#include "kernels/adaptive_thresholding.cuh"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "You must run this program as: bradley <path to image>";
        return 1;
    }
    std::cout << "Bradley algorithm on image " << argv[1] << std::endl;
    cv::Mat srcImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    unsigned char *imgArray = srcImg.isContinuous() ? srcImg.data : srcImg.clone().data;
    unsigned int imgSize = srcImg.total() * srcImg.channels();
    unsigned int bufferSize = (srcImg.cols+1)+(srcImg.rows+1);

    std::cout << "width: " << srcImg.cols << std::endl
              << "height: " << srcImg.rows << std::endl;
    unsigned char *inputArrayGPU = allocateArrayOnGPU<unsigned char>(imgSize);
    transferDataToGPU<unsigned char>(inputArrayGPU, imgArray, imgSize);
    unsigned char *bufferArrayGPU = allocateArrayOnGPU<unsigned char>(bufferSize);
    setMemoryOnGPU<unsigned char>(bufferArrayGPU, 0, imgSize);
    unsigned char *outputArrayGPU = allocateArrayOnGPU<unsigned char>(imgSize);
    setMemoryOnGPU<unsigned char>(outputArrayGPU, 0, imgSize);

    unsigned long threadNum = 128;
    unsigned long blocksNum = 256;
    bradleyBinarization<<< blocksNum, threadNum >>>(inputArrayGPU, bufferArrayGPU, outputArrayGPU, srcImg.cols,
                                                    srcImg.rows);
    synchronizeKernel();

    transferDataFromGPU<unsigned char>(imgArray, outputArrayGPU, imgSize);
    cudaFree(inputArrayGPU);

    cv::imwrite("/home/students/2021DS/grkrol/prir/out.jpeg", srcImg);
    return 0;
}
