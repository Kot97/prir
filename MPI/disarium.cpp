#include <iostream>
#include <mpi.h>

#include "lib/benchmarking.hpp"
#include "lib/disarium_number.hpp"

int main(int argc, char *argv[])
{
    std::size_t size = 100000;
    char *result = (char*)malloc(sizeof(char) * size);
    auto timing_function = [](){ return MPI_Wtime(); };

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double mpi_time = benchmark(1000, timing_function, [size, result]()
                                    { generate_disarium_numbers_mpi(result, size, MPI_COMM_WORLD); });

    if(rank == 0) std::cout << "MPI time: " << mpi_time << std::endl;
    free(result);
    MPI_Finalize();
}
