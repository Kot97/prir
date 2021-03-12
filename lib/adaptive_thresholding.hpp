#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>

class adaptive_thresholding
{
    cv::Mat image;

public:
    adaptive_thresholding(const std::string& image_path) : image(cv::imread(image_path, cv::IMREAD_GRAYSCALE)) {}
    cv::Mat run_serial();
    cv::Mat run_openmp(std::size_t threads_count);
};