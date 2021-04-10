#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "mpi.h"

#include <string>

void run_serial(const cv::Mat&, cv::Mat&);
void run_mpi(const cv::Mat&, cv::Mat&, MPI_Comm);
