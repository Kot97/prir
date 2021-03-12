#pragma once

#include <cstddef>

template <typename TimingFunction, typename Function>
decltype(auto) benchmark(std::size_t iterations, TimingFunction timing_function, Function benchmarked_function)
{
    decltype(timing_function()) result = 0;

    for(int i = 0; i < iterations; i++)
    {
        ResultType start = timing_function();
        benchmarked_function();
        result += timing_function() - start;
    }

    return result / iterations;
}
