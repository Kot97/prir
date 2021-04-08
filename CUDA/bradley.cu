#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "cudaUtils.cuh"
#include "kernels/adaptive_thresholding.cuh"

const unsigned int THREADS_NUM = 128;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "You must run this program as: bradley <path to image>";
        return 1;
    }
    std::cout << "Bradley algorithm on image " << argv[1] << std::endl;
    cv::Mat srcImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    unsigned char *imgArray = srcImg.isContinuous() ? srcImg.data : srcImg.clone().data;
    unsigned int imgSize = srcImg.total() * srcImg.channels();

    std::cout << "width: " << srcImg.cols << std::endl
              << "height: " << srcImg.rows << std::endl;

    unsigned char *inputArrayGPU = allocateArrayOnGPU<unsigned char>(imgSize);
    transferDataToGPU<unsigned char>(inputArrayGPU, imgArray, imgSize);

    unsigned int *bufferArrayGPU = allocateArrayOnGPU<unsigned int>(imgSize);

    unsigned char *outputArrayGPU = allocateArrayOnGPU<unsigned char>(imgSize);
//    setMemoryOnGPU<unsigned char>(outputArrayGPU, 0, imgSize);


    unsigned long maxSize = max(srcImg.cols, srcImg.rows);

    unsigned long blocksNum = getBlocksNumber(THREADS_NUM, maxSize);

    std::cout << "THREADS_NUM: " << THREADS_NUM << std::endl
              << "blocksNum: " << blocksNum << std::endl;
    std::cout << "unsigned char: " << sizeof(unsigned char)<< std::endl
              << "unsigned int: " << sizeof(unsigned int) << std::endl;

    bradleyBinarization<<< blocksNum, THREADS_NUM >>>(inputArrayGPU, bufferArrayGPU, outputArrayGPU, srcImg.cols,
                                                      srcImg.rows);
    synchronizeKernel();

    transferDataFromGPU<unsigned char>(imgArray, outputArrayGPU, imgSize);

    cudaFree(inputArrayGPU);
    cudaFree(bufferArrayGPU);
    cudaFree(outputArrayGPU);

    cv::imwrite("/home/students/2021DS/grkrol/prir/out.jpeg", srcImg);
    return 0;
}
