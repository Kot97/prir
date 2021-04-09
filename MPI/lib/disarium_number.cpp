#include "disarium_number.hpp"

#include <cmath>
#include <iostream> // debug

unsigned int count_digits(unsigned int number)
{
    unsigned int digits_count = 0;

    while (number)
    {
        number /= 10;
        digits_count++;
    }

    return digits_count;
}

bool is_number_disarium(unsigned int number)
{
    unsigned int sum = 0, temp = number;
    unsigned int digits_count = count_digits(number);

    while (temp)
    {
        sum += std::pow(temp % 10, digits_count--);
        temp /= 10;
    }

    return sum == number;
}

void generate_disarium_numbers_serial(char* result, std::size_t end, std::size_t begin)
{
    char *ptr = result;

    for (std::size_t i = begin; i < end - begin; ++i, ++ptr)
        if (is_number_disarium(i))
            *ptr = true;
        else
            *ptr = false;
}

void rank0(char* result, std::size_t size, MPI_Comm comm)
{
    int num;
    MPI_Comm_size(comm, &num);
    int increment = size / num;
    std::size_t rest = size % num;

    MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request) * (num-1));

    for (int i = 1; i < num; i++)
        MPI_Irecv(result + rest + increment * i, increment, MPI_CHAR, i, MPI_ANY_TAG, comm, &req[i-1]);

    generate_disarium_numbers_serial(result, increment + rest);

    MPI_Waitall(num-1, req, MPI_STATUS_IGNORE);
    free(req);
}

void others(std::size_t size, MPI_Comm comm)
{
    int num, rank;
    MPI_Comm_size(comm, &num);
    MPI_Comm_rank(comm, &rank);

    int increment = size / num;
    std::size_t increment_rest = increment + size % num;

    char *buff = (char*)malloc(sizeof(char) * increment);
    generate_disarium_numbers_serial(buff, increment_rest + increment * (rank + 1), increment_rest + increment * rank);
    MPI_Send(buff, increment, MPI_CHAR, 0, 0, comm);

    free(buff);
}

void generate_disarium_numbers_mpi(char* result, std::size_t size, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) rank0(result, size, comm);
    else others(size, comm);
}
