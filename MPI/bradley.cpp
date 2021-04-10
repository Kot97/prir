#include <iostream>
#include <mpi.h>

#include "lib/adaptive_thresholding.hpp"
#include "lib/benchmarking.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Bradley algorithm on image " << argv[3] << std::endl;
    adaptive_thresholding algorithm(argv[3]);
    auto timing_function = [](){ return MPI_Wtime(); };

    MPI_Init(&argc, &argv);
    double mpi_time = benchmark(40, timing_function, [&algorithm](){ algorithm.run_mpi(MPI_COMM_WORLD); });
    std::cout << "MPI time: " << mpi_time << std::endl;
    MPI_Finalize();
}
