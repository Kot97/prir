__device__ unsigned long countDigits(unsigned long number);

__device__ bool isNumberDisarium(unsigned long number);

__device__ unsigned long pow(unsigned long x,unsigned long n);

__device__ void addResult(unsigned long *generatedNumbersGPU, unsigned long result);

__global__ void generateDisariumNumbers(unsigned long *generatedNumbersGPU, const unsigned long NUMBERS_COUNT) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < NUMBERS_COUNT) {
        if (isNumberDisarium(i))
            addResult(generatedNumbersGPU, i);
        i += blockDim.x*gridDim.x;
    }
}

__device__ void addResult(unsigned long *generatedNumbersGPU, unsigned long result) {

}

__device__ bool isNumberDisarium(unsigned long number) {
    unsigned long sum = 0, temp = number;
    unsigned long digitsCount = countDigits(number);
    while (temp) {
        sum += pow(temp % 10, digitsCount--);
        temp /= 10;
    }
    return sum == number;
}

__device__ unsigned long countDigits(unsigned long number) {
    unsigned int digits_count = 0;
    while (number) {
        number /= 10;
        digits_count++;
    }
    return digits_count;
}

__device__ unsigned long pow(unsigned long x, unsigned long n) {
    unsigned long result=1;
    for (unsigned int i = 0; i < n; i++)
        result*=x;
    return result;
}

