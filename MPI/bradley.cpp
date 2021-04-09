#include <iostream>
#include <mpi.h>

#include "lib/adaptive_thresholding.hpp"
#include "lib/benchmarking.hpp"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "You must run this program as: mpirun -np <nodes count> bradley <path to image>";
        return 1;
    }

    std::cout << "Bradley algorithm on image " << argv[3] << std::endl;

    adaptive_thresholding algorithm(argv[3]);
    auto timing_function = [](){ return MPI_Wtime(); };

    double serial_time = benchmark(20, timing_function, [&algorithm](){ algorithm.run_serial(); });
    std::cout << "Serial time: " << serial_time << std::endl;

    MPI_Init(&argc, &argv);
    double mpi_time = benchmark(20, timing_function, [&algorithm](){ algorithm.run_mpi(MPI_COMM_WORLD); });
    std::cout << "MPI time: " << mpi_time << std::endl;
    MPI_Finalize();
}
