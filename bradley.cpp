#include <iostream>
#include <omp.h>

#include "lib/adaptive_thresholding.hpp"
#include "lib/benchmarking.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "You must run this program as: bradley <threads count> <path to image>";
        return 1;
    }

    std::cout << "Bradley algorithm on image " << argv[2] << std::endl;

    adaptive_thresholding algorithm(argv[2]);
    auto timing_function = [](){ return omp_get_wtime(); };

    int serial_time = benchmark(20, timing_function, [&algorithm](){ algorithm.run_serial(); });
    int openmp_time = benchmark(20, timing_function,
                                [&algorithm, argv](){ algorithm.run_openmp(strtol(argv[1], nullptr, 10)); });

    std::cout << "Serial time: " << serial_time << std::endl;
    std::cout << "OpenMP time: " << openmp_time << std::endl;

    return 0;
}
