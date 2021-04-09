#include <iostream>
#include <mpi.h>

#include "lib/benchmarking.hpp"
#include "lib/disarium_number.hpp"

int main(int argc, char *argv[])
{
    std::size_t size = 100000;
    char *result = new char[size];
    auto timing_function = [](){ return MPI_Wtime(); };

    double serial_time = benchmark(1000, timing_function, [size, result]()
                                    { generate_disarium_numbers_serial(result, size); });
    std::cout << "Serial time: " << serial_time << std::endl;
    delete[] result;
}
