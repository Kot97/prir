#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "cudaUtils.cuh"
#include "kernels/adaptive_thresholding.cuh"

const unsigned int THREADS_NUM = 64;

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

    unsigned int S = srcImg.cols/8;
    unsigned int s2 = S/2;
    double t = 15;
    checkError(cudaMemcpyToSymbol(S2, &s2, sizeof(S2)), "error during copy data to symbol S on GPU");
    checkError(cudaMemcpyToSymbol(T, &t, sizeof(T)), "error during copy data to symbol T on GPU");
    checkError(cudaMemcpyToSymbol(width, &srcImg.cols, sizeof(width)), "error during copy data to symbol width on GPU");
    checkError(cudaMemcpyToSymbol(height, &srcImg.rows, sizeof(height)), "error during copy data to symbol height on GPU");

    unsigned long maxSize = max(srcImg.cols, srcImg.rows);

    unsigned long blocksNum = getBlocksNumber(THREADS_NUM, maxSize);

    std::cout << "THREADS_NUM: " << THREADS_NUM << std::endl
              << "blocksNum: " << blocksNum << std::endl;

    bradleyBinarization<<< blocksNum, THREADS_NUM >>>(inputArrayGPU, bufferArrayGPU, outputArrayGPU);
    synchronizeKernel();

    transferDataFromGPU<unsigned char>(imgArray, outputArrayGPU, imgSize);

    cudaFree(inputArrayGPU);
    cudaFree(bufferArrayGPU);
    cudaFree(outputArrayGPU);

    cv::imwrite("/home/students/2021DS/grkrol/prir/out.jpeg", srcImg); //todo fix path
    return 0;
}
