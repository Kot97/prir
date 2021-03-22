__device__ unsigned int countDigits(unsigned long number);

__device__ bool isNumberDisarium(unsigned long number);

__device__ unsigned long pow(unsigned long x, unsigned int n);

__device__ void addResult(unsigned long *generatedNumbersGPU, unsigned long result);

__device__ unsigned int findedNumbersCount = 0;//todo ogarnac inicjalizacje

__global__ void generateDisariumNumbers(unsigned long *generatedNumbersGPU, const unsigned int NUMBERS_COUNT) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    while (findedNumbersCount < NUMBERS_COUNT) {
        if (isNumberDisarium(i))
            addResult(generatedNumbersGPU, i);
        i += blockDim.x * gridDim.x;
    }
}

__device__ void addResult(unsigned long *generatedNumbersGPU, unsigned long result) {
    generatedNumbersGPU[atomicAdd(&findedNumbersCount, 1)] = result;//todo test synchronization
}

__device__ bool isNumberDisarium(unsigned long number) {
    unsigned long sum = 0, temp = number;
    unsigned int digitsCount = countDigits(number);
    while (temp) {
        sum += pow(temp % 10, digitsCount--);
        temp /= 10;
    }
    return sum == number;
}

__device__ unsigned int countDigits(unsigned long number) {
    unsigned int digitsCount = 0;
    while (number) {
        number /= 10;
        digitsCount++;
    }
    return digitsCount;
}

__device__ unsigned long pow(unsigned long x, unsigned int n) {
    unsigned long result = 1;
    for (unsigned int i = 0; i < n; i++)
        result *= x;
    return result;
}