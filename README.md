# KNN Classifier for MNIST Dataset

This repository contains a C++ implementation of a K-Nearest Neighbors (KNN) classifier for the MNIST dataset of handwritten digits.

## Dependencies

- Qt5
- OpenMP

## Building

To build the project, use the following commands:

```bash
mkdir build
cd build
qmake ..
make
```

## Running

To run the program, use the following command:

```bash
./your_program
```

Replace `your_program` with the name of your program.

You can specify the number of threads to use with OpenMP by setting the `OMP_NUM_THREADS` environment variable:

```bash
export OMP_NUM_THREADS=4
./your_program
```

## Usage

The program loads the MNIST training data from a CSV file, trains the KNN classifier, and then tests the classifier on the MNIST test data. The results are printed to the console.

You can also use the program to classify your own images of handwritten digits. Draw a digit in the window that appears when you run the program, and the program will classify the digit and print the result to the console.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
