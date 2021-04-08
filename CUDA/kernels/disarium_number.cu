__device__ unsigned int countDigits(unsigned int number);

__device__ bool isNumberDisarium(unsigned int number);

__device__ unsigned int pow(unsigned int x, unsigned int n);

__global__ void generateDisariumNumbers(unsigned int *generatedNumbers, bool *result, const unsigned int NUMBERS_COUNT) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUMBERS_COUNT; i += blockDim.x)
        result[i] = isNumberDisarium(generatedNumbers[i]);
}

__device__ bool isNumberDisarium(unsigned int number) {
    unsigned int sum = 0, temp = number;
    unsigned int digitsCount = countDigits(number);
    while (temp) {
        sum += pow(temp % 10, digitsCount--);
        temp /= 10;
    }
    return sum == number;
}

__device__ unsigned int countDigits(unsigned int number) {
    unsigned int digitsCount = 0;
    while (number) {
        number /= 10;
        digitsCount++;
    }
    return digitsCount;
}

__device__ unsigned int pow(unsigned int x, unsigned int n) {
    unsigned int result = 1;
    for (unsigned int i = 0; i < n; i++)
        result *= x;
    return result;
}