// insert header guards here
#pragma once

#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <vector>
#include <queue>
#include <iomanip>

#include <cstdio>
#include <stdexcept>
#include <cstdarg>
#include <omp.h>
#include "utils.hpp"

/**
 * @brief tempalted class for the data
 *
 * @tparam K k nearest neighbour
 * @tparam N the training data size should be NxN
 * @tparam M number of training data set where we should have M data sets of NxN size
 */
template <int K, int N, int M>
class KNN
{
private:
    // X value of the training data
    uint8_t X[M][N * N];
    // Y value of the training data
    uint8_t Y[M];

    // Y values of the test data
    std::vector<uint8_t> YTest;

    // confusion matrix
    int ConfusionMatrix[10][10] = {0};

public:
    KNN(std::string TrainingFile)
    {
        // open the csv file and read the data b;ased on the provided format
        std::ifstream file(TrainingFile);
        ASSERT(file.is_open(), "Error: file not found");

        std::string line;
        int lineCount = 0;
        // skip the first line
        std::getline(file, line);
        while (std::getline(file, line))
        {
            // std::cout << line << std::endl;
            // split the line based on the comma
            std::stringstream ss(line);
            std::string token;
            int RowCount = 0;

            // handle the label of the data
            std::getline(ss, token, ',');
            Y[lineCount] = static_cast<uint8_t>(std::stoi(token));

            // handle the rest
            while (std::getline(ss, token, ','))
            {
                // read the data and store it in the X and Y
                X[lineCount][RowCount++] = static_cast<uint8_t>(std::stoi(token));

                // std::cout << "RowCount: " << RowCount << std::endl;
            }
            // std::cout << "lineCount: " << lineCount << std::endl;
            lineCount++;
            ASSERT(RowCount == N * N, "Error: Invalid data size in N * N with %d", RowCount);
        }
        ASSERT(lineCount == M, "Error: Invalid data size in M*M with %d", lineCount);

        // close the file
        file.close();

        fprintf(stdout, "KNN::finished training \n");
    }

    // northing to do in the destructor
    ~KNN() {}

    // function to predict
    std::vector<uint8_t> test(std::string aImageFile)
    {
        // has to use array to avoid the resizing
        std::vector<std::array<uint8_t, N * N>> XTest;
        std::vector<uint8_t> YPredicted;

        // assume the png file is given size and it matches the training data size
        // open the file and read the data
        std::ifstream file(aImageFile);
        ASSERT(file.is_open(), "Error: file not found");

        // number of data lines
        int numLines = std::count(std::istreambuf_iterator<char>(file),
                                  std::istreambuf_iterator<char>(), '\n') -
                       1;

        file.clear();                 // Clear any error flags
        file.seekg(0, std::ios::beg); // Reset read position to beginning of file

        XTest.resize(numLines);
        YTest.resize(numLines);
        YPredicted.resize(numLines);

        int lineCount = 0;
        std::string line;
        std::getline(file, line);

        // process the rest
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string token;

            // handle the label of the data
            std::getline(ss, token, ',');
            YTest.at(lineCount) = static_cast<uint8_t>(std::stoi(token));

            int RowCount = 0;
            while (std::getline(ss, token, ','))
            {
                XTest[lineCount][RowCount++] = static_cast<uint8_t>(std::stoi(token));
            }
            ASSERT(RowCount == N * N, "Error: Invalid data size");
            lineCount++;
        }
        ASSERT(lineCount == numLines, "Error: Invalid data size in line count %d and num lines is %d", lineCount, numLines);

        // close the file
        file.close();

        fprintf(stdout, "KNN::finished reading testing \n");

        // Now perform the computation for every data set in training find the euclean distance and store the distance and the label
        std::vector<int[M]> distances(numLines);

#pragma omp parallel for
        for (int iTest = 0; iTest < numLines; iTest++)
        {
            for (int i = 0; i < M; i++)
            {
                // find the euclean distance
                int distance = 0;
                distance = std::inner_product(X[i], X[i] + N * N, XTest[iTest].data(), 0, std::plus<>(), [](uint8_t a, uint8_t b) -> int
                                              {
                                                  return int(a - b) * int(a - b);
                                                  // store the distance and the label
                                              });
                // std::cout << "distance: " << distance << std::endl;
                distances[iTest][i] = distance;
            }

            // convert it into a vector and sort it out
            std::vector<int> distances_vector(std::begin(distances[iTest]), std::end(distances[iTest]));
            std::vector<int> indices = find_min_k_indices(distances_vector, K);

            // find the values of the Y for the indices and do a majority vote
            // use bucket sort to detemine the values , indices are the labels while values are the count
            // each bucket has value of zero
            int Votes[10] = {0};
            // count the votes
            for (int i = 0; i < K; i++)
            {
                Votes[Y[indices[i]]]++;
            }

            auto MaxElement = std::max_element(Votes, Votes + 10);
            int Index = std::distance(Votes, MaxElement);
            YPredicted[iTest] = static_cast<uint8_t>(Index);
        }

#pragma omp parallel for
        for (int i = 0; i < numLines; i++)
        {
            ConfusionMatrix[YTest[i]][YPredicted[i]]++;
        }

        return YPredicted;
    }

    // predict
    uint8_t
    predict(uint8_t tImage[N * N])
    {
        int distances[M];

        for (int i = 0; i < M; i++)
        {
            // find the euclean distance
            int distance = 0;
            distance = std::inner_product(X[i], X[i] + N * N, tImage, 0, std::plus<>(), [](uint8_t a, uint8_t b) -> int
                                          {
                                              return int(a - b) * int(a - b);
                                              // store the distance and the label
                                          });
            distances[i] = distance;
        }

        // convert it into a vector and sort it out
        std::vector<int> distances_vector(std::begin(distances), std::end(distances));
        std::vector<int> indices = find_min_k_indices(distances_vector, K);

        // find the values of the Y for the indices and do a majority vote
        // use bucket sort to determine the values , indices are the labels while values are the count
        // each bucket has value of zero
        int Votes[10] = {0};
        // count the votes
        for (int i = 0; i < K; i++)
        {
            Votes[Y[indices[i]]]++;
        }

        auto MaxElement = std::max_element(Votes, Votes + 10);
        int Index = std::distance(Votes, MaxElement);
        return static_cast<uint8_t>(Index);
    }

    void
    print_data_set(int MM = M)
    {
        for (int i = 0; i < MM; i++)
        {
            std::cout << "Label is: " << static_cast<int>(Y[i]) << std::endl;
            std::for_each(X[i], X[i] + N * N, [](const auto &elem)
                          { std::cout << static_cast<int>(elem) << ' '; });

            std::cout << std::endl;
        }
    }

    // ----------------- getter functions -----------------
    std::vector<uint8_t> const &
    get_y_test() const
    {
        return YTest;
    }

    /**
     * @brief print the confusion matrix
     *
     */
    void
    print_confusion_matrix_fancy()
    {
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                std::cout << std::setw(4) << ConfusionMatrix[i][j] << " ";
            }
            std::cout << " ... " << std::endl;
        }
    }
};
