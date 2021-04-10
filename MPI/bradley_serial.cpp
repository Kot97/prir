#include <iostream>
#include <mpi.h>

#include "lib/adaptive_thresholding.hpp"
#include "lib/benchmarking.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Bradley algorithm on image " << argv[1] << std::endl;
    adaptive_thresholding algorithm(argv[1]);
    auto timing_function = [](){ return MPI_Wtime(); };

    double serial_time = benchmark(40, timing_function, [&algorithm](){ algorithm.run_serial(); });
    std::cout << "Serial time: " << serial_time << std::endl;
}
