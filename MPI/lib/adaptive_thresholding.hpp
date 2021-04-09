#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "mpi.h"

#include <string>

class adaptive_thresholding
{
    cv::Mat image;

public:
    explicit adaptive_thresholding(const std::string& image_path) : image(cv::imread(image_path, cv::IMREAD_GRAYSCALE)) {}
    cv::Mat run_serial();
    cv::Mat run_mpi(MPI_Comm);

private:
    void run_mpi_rank0(MPI_Comm, cv::Mat&);
    void run_mpi_others_rank(MPI_Comm);
};