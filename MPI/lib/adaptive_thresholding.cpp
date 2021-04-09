#include "adaptive_thresholding.hpp"

#include <opencv2/imgproc/imgproc.hpp>

cv::Mat adaptive_thresholding::run_serial()
{
    cv::Mat sumMat, result = cv::Mat::zeros(image.size(), CV_8UC1);

    int rows = image.rows;
    int cols = image.cols;

    int s = MAX(rows, cols) / 16;
    double T = 0.15;

    uchar *p_inputMat, *p_outputMat;
    int x1, y1, x2, y2, count, sum;
    int *p_y1, *p_y2;

    cv::integral(image, sumMat);

    for (int i = 0; i < rows; ++i)
    {
        y1 = i - s;
        y2 = i + s;

        if (y1 < 0) y1 = 0;
        if (y2 >= rows) y2 = rows - 1;

        p_y1 = sumMat.ptr<int>(y1);
        p_y2 = sumMat.ptr<int>(y2);
        p_inputMat = image.ptr<uchar>(i);
        p_outputMat = result.ptr<uchar>(i);

        for (int j = 0; j < cols; ++j)
        {
            x1 = j - s;
            x2 = j + s;

            if (x1 < 0) x1 = 0;
            if (x2 >= cols) x2 = cols - 1;

            count = (x2 - x1) * (y2 - y1);

            // I(x,y)= s(x2,y2) - s(x1,y2) - s(x2,y1) + s(x1,x1)
            sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

            if (static_cast<int>(p_inputMat[j] * count) < static_cast<int>(sum * (1.0 - T)))
                p_outputMat[j] = 255;
            else
                p_outputMat[j] = 0;
        }
    }

    return result;
}

cv::Mat adaptive_thresholding::run_mpi(MPI_Comm comm)
{
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC1);
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) run_mpi_rank0(comm, result);
    else run_mpi_others_rank(comm);

    return result;
}

void adaptive_thresholding::run_mpi_rank0(MPI_Comm comm, cv::Mat& result)
{
    cv::Mat sumMat;

    int rows = image.rows;
    int cols = image.cols;

    int s = MAX(rows, cols) / 16;
    double T = 0.15;

    cv::integral(image, sumMat);

    for (int i = 0; i < rows; ++i)
    {
        uchar *p_inputMat, *p_outputMat;
        int *p_y1, *p_y2;
        int y1, y2;

        y1 = i - s;
        y2 = i + s;

        if (y1 < 0) y1 = 0;
        if (y2 >= rows) y2 = rows - 1;

        p_y1 = sumMat.ptr<int>(y1);
        p_y2 = sumMat.ptr<int>(y2);
        p_inputMat = image.ptr<uchar>(i);
        p_outputMat = result.ptr<uchar>(i);

        for (int j = 0; j < cols; ++j)
        {
            int x1, x2, count, sum;

            x1 = j - s;
            x2 = j + s;

            if (x1 < 0) x1 = 0;
            if (x2 >= cols) x2 = cols-1;

            count = (x2 - x1) * (y2 - y1);

            // I(x,y)= s(x2,y2) - s(x1,y2) - s(x2,y1) + s(x1,x1)
            sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

            if (static_cast<int>(p_inputMat[j] * count) < static_cast<int>(sum * (1.0 - T)))
                p_outputMat[j] = 255;
            else
                p_outputMat[j] = 0;
        }
    }
}

void adaptive_thresholding::run_mpi_others_rank(MPI_Comm comm)
{
}