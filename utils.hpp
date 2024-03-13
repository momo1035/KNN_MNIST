#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <stdarg.h>
#include <queue>
#include <algorithm>

void AssertHandler(const char *condition, const char *filename, int line, const char *format, ...)
{
    va_list args;
    va_start(args, format);

    fprintf(stderr, "Assertion failed: (%s), file %s, line %d. ", condition, filename, line);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");

    va_end(args);

    throw std::runtime_error("Assertion failed");
}

#define ASSERT(condition, format, ...)                                        \
    if (!(condition))                                                         \
    {                                                                         \
        AssertHandler(#condition, __FILE__, __LINE__, format, ##__VA_ARGS__); \
    }

// forward declaration of the function
std::vector<int> find_min_k_indices(const std::vector<int> &distances, int k);

void print_as_row_vector(const std::vector<int> &vec)
{
    for (const auto &elem : vec)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;
}

std::vector<int> find_min_k_indices(const std::vector<int> &distances, int k)
{
    using Pair = std::pair<int, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::less<Pair>> pq;

    for (int i = 0; i < distances.size(); ++i)
    {
        if (pq.size() < k)
        {
            pq.push({distances[i], i});
        }
        else if (distances[i] < pq.top().first)
        {
            pq.pop();
            pq.push({distances[i], i});
        }
    }

    std::vector<int> indices;
    while (!pq.empty())
    {
        indices.push_back(pq.top().second);
        pq.pop();
    }

    std::reverse(indices.begin(), indices.end());
    return indices;
}