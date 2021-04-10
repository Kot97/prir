#include <iostream>
#include <mpi.h>

#include "lib/adaptive_thresholding.hpp"
#include "lib/benchmarking.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::cout << "Bradley algorithm on image " << argv[1] << std::endl;

    cv::Mat image(cv::imread(argv[1], cv::IMREAD_GRAYSCALE));
    if (rank == 0)
    {
        std::cout << "Image size: " << image.size() << std::endl;
    }
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC1);
    auto timing_function = [](){ return MPI_Wtime(); };

    double mpi_time = benchmark(40, timing_function, [&image, &result](){ run_mpi(image, result, MPI_COMM_WORLD); });
    if (rank == 0) std::cout << "MPI time: " << mpi_time << std::endl;
    MPI_Finalize();
}
