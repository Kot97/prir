#pragma once

#include <vector>
#include "mpi.h"

bool is_number_disarium(unsigned int number);
void generate_disarium_numbers_serial(char * result, std::size_t end, std::size_t begin = 0);
void generate_disarium_numbers_mpi(char *result, std::size_t size, MPI_Comm comm);
