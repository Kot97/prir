#include <iostream>
#include <mpi.h>

#include "lib/adaptive_thresholding.hpp"
#include "lib/benchmarking.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Bradley algorithm on image " << argv[1] << std::endl;
    cv::Mat image(cv::imread(argv[1], cv::IMREAD_GRAYSCALE));
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC1);
    auto timing_function = [](){ return MPI_Wtime(); };

    double serial_time = benchmark(40, timing_function, [&image, &result](){ run_serial(image, result); });
    std::cout << "Serial time: " << serial_time << std::endl;
}
