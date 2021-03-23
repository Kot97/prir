#ifndef PRIR_CUDAUTILS_CUH
#define PRIR_CUDAUTILS_CUH

void synchronizeKernel() {
    cudaError_t errorCode = cudaDeviceSynchronize();
    if (errorCode != cudaSuccess) {
        std::cout << "error during Device Synchronize: " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<typename T>
void transferDataFromGPU(T *generatedNumbersCPU, T *generatedNumbersGPU, unsigned long elementNumber) {
    cudaError_t errorCode = cudaMemcpy(generatedNumbersCPU, generatedNumbersGPU, sizeof(T) * elementNumber,
                                       cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess) {
        std::cout << "error during transfer data from gpu " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<typename T>
void transferDataToGPU(T *generatedNumbersGPU, T *generatedNumbersCPU, unsigned long elementNumber) {
    cudaError_t errorCode = cudaMemcpy(generatedNumbersGPU, generatedNumbersCPU, sizeof(T) * elementNumber,
                                       cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess) {
        std::cout << "error during transfer data to gpu " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}
template<typename T>
T *allocateArrayOnGPU(const unsigned long elementsNumber) {
    T *table_addr;
    cudaError_t errorCode = cudaMalloc((void **) &table_addr, elementsNumber * sizeof(T));
    if (errorCode != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    return table_addr;
}

#endif //PRIR_CUDAUTILS_CUH
