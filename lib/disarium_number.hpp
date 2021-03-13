#pragma once

#include <vector>

bool is_number_disarium(unsigned int number);
std::vector<bool> generate_disarium_numbers_serial(std::size_t size);
std::vector<bool> generate_disarium_numbers_openmp(std::size_t size, std::size_t thread_count);
