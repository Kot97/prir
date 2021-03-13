#include <iostream>
#include <omp.h>

#include "lib/benchmarking.hpp"
#include "lib/disarium_number.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "You must run this program as: disarium <threads count>";
        return 1;
    }

    std::cout << "Disarium number algorithm for thread count: " << argv[1] << std::endl;

    std::size_t size = 100000;
    auto timing_function = [](){ return omp_get_wtime(); };

    double serial_time = benchmark(100, timing_function, [size](){ generate_disarium_numbers_serial(size); });
    double openmp_time = benchmark(100, timing_function,
                                [size, argv](){ generate_disarium_numbers_openmp(size, strtol(argv[1], nullptr, 10)); });

    std::cout << "Serial time: " << serial_time << std::endl;
    std::cout << "OpenMP time: " << openmp_time << std::endl;

    return 0;
}
