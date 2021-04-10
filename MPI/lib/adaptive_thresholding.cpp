#include "adaptive_thresholding.hpp"

#include <opencv2/imgproc/imgproc.hpp>

void run_serial(const cv::Mat& image, cv::Mat& result)
{
    cv::Mat sumMat;

    int rows = image.rows;
    int cols = image.cols;

    int s = MAX(rows, cols) / 16;
    double T = 0.15;

    const uchar *p_inputMat;
    uchar *p_outputMat;
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
}

void rank0(const cv::Mat& image, cv::Mat& result, MPI_Comm comm)
{
    int num;
    MPI_Comm_size(comm, &num);
    int increment = image.rows / num;
    int rest = image.rows % num;

    MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request) * (num-1));

    for (int i = 1; i < num; i++)
        MPI_Irecv(result.data + rest + increment * i * image.cols, increment * image.cols, MPI_CHAR, i, MPI_ANY_TAG, comm, &req[i-1]);

    run_serial(cv::Mat(increment + rest, image.cols, CV_8UC1, (void*)image.data), result);

    MPI_Waitall(num-1, req, MPI_STATUS_IGNORE);
    free(req);
}

void others(const cv::Mat& image, MPI_Comm comm)
{
    int num, rank;
    MPI_Comm_size(comm, &num);
    MPI_Comm_rank(comm, &rank);

    int increment = image.rows / num;
    int rest = image.rows % num;

    cv::Mat result = cv::Mat::zeros(increment, image.cols, CV_8UC1);

    run_serial(cv::Mat(increment, image.cols, CV_8UC1, (void*)(image.data + image.cols * (rest + increment * rank ))), result);
    MPI_Send(result.data, increment * image.cols, MPI_CHAR, 0, 0, comm);
}

void run_mpi(const cv::Mat& image, cv::Mat& result, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) rank0(image, result, comm);
    else others(image, comm);
}
