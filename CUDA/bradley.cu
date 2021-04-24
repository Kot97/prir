#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <channel_descriptor.h>
#include "cudaUtils.cuh"
#include "kernels/adaptive_thresholding.cuh"

const unsigned int THREADS_NUM = 64;

cv::Mat readImage(int argc, char *const *argv);

void runKernels(const unsigned char *inputArrayGPU, unsigned int *bufferArrayGPU, unsigned char *outputArrayGPU, unsigned int cols,
                unsigned int rows);

void setConstantMemory(unsigned int cols, unsigned int rows);

int main(int argc, char **argv) {
    cv::Mat srcImg = readImage(argc, argv);

    unsigned char *imgArray = srcImg.isContinuous() ? srcImg.data : srcImg.clone().data;
    unsigned int imgSize = srcImg.cols * srcImg.rows;

    unsigned char *inputArrayGPU = allocateArrayOnGPU<unsigned char>(imgSize);
    transferDataToGPU<unsigned char>(inputArrayGPU, imgArray, imgSize);
    unsigned int *bufferArrayGPU = allocateArrayOnGPU<unsigned int>(imgSize);
    unsigned char *outputArrayGPU = allocateArrayOnGPU<unsigned char>(imgSize);

    setConstantMemory(srcImg.cols, srcImg.rows);
    runKernels(inputArrayGPU, bufferArrayGPU, outputArrayGPU, srcImg.cols, srcImg.rows);

    transferDataFromGPU<unsigned char>(imgArray, outputArrayGPU, imgSize);

    freeArrayGPU(inputArrayGPU);
    freeArrayGPU(bufferArrayGPU);
    freeArrayGPU(outputArrayGPU);

    cv::imwrite("/home/students/2021DS/grkrol/prir/out.jpeg", srcImg); //todo fix path
    return 0;
}

cv::Mat readImage(int argc, char *const *argv) {
    if (argc != 2) {
        std::cerr << "You must run this program as: bradley <path to image>";
        exit(EXIT_FAILURE);
    }
    std::cout << "Bradley algorithm on image " << argv[1] << std::endl;
    cv::Mat srcImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    return srcImg;
}

void setConstantMemory(unsigned int cols, unsigned int rows) {
    unsigned int S = cols / 8;
    unsigned int s2 = S / 2;
    double t = 0.15;
    std::cout << "s2: " << s2 << std::endl
              << "t: " << t << std::endl;
    checkError(cudaMemcpyToSymbol(S2, &s2, sizeof(S2)), "error during copy data to symbol S on GPU");
    checkError(cudaMemcpyToSymbol(T, &t, sizeof(T)), "error during copy data to symbol T on GPU");
    checkError(cudaMemcpyToSymbol(width, &cols, sizeof(width)), "error during copy data to symbol width on GPU");
    checkError(cudaMemcpyToSymbol(height, &rows, sizeof(height)), "error during copy data to symbol height on GPU");
}

void runKernels(const unsigned char *inputArrayGPU, unsigned int *bufferArrayGPU, unsigned char *outputArrayGPU, unsigned int cols,
                unsigned int rows) {
    unsigned long blocksNum;
    //todo add benchmarks
    blocksNum = getBlocksNumber(THREADS_NUM, cols);
    verticalSum<<< blocksNum, THREADS_NUM >>>(inputArrayGPU, bufferArrayGPU);
    synchronizeKernel();

    blocksNum = getBlocksNumber(THREADS_NUM, rows);
    horizontalSum<<< blocksNum, THREADS_NUM >>>(bufferArrayGPU);
    synchronizeKernel();

    blocksNum = getBlocksNumber(THREADS_NUM, cols * rows);
    binarization<<< blocksNum, THREADS_NUM >>>(inputArrayGPU, bufferArrayGPU, outputArrayGPU);
    synchronizeKernel();
}
