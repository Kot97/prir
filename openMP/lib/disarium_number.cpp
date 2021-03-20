#include "disarium_number.hpp"

#include <cmath>

unsigned int count_digits(unsigned int number)
{
    unsigned int digits_count = 0;

    while (number)
    {
        number /= 10;
        digits_count++;
    }

    return digits_count;
}

bool is_number_disarium(unsigned int number)
{
    unsigned int sum = 0, temp = number;
    unsigned int digits_count = count_digits(number);

    while (temp)
    {
        sum += std::pow(temp % 10, digits_count--);
        temp /= 10;
    }

    return sum == number;
}

std::vector<bool> generate_disarium_numbers_serial(std::size_t size)
{
    std::vector<bool> result(size, false);
    auto ptr = result.begin();

    for (int i = 0; i < size; ++i, ++ptr)
        if (is_number_disarium(i))
            *ptr = true;

    return result;
}

std::vector<bool> generate_disarium_numbers_openmp(std::size_t size, std::size_t thread_count)
{
    std::vector<bool> result(size, false);

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < size; ++i)
        if (is_number_disarium(i))
            result[i] = true;

    return result;
}
