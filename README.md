
# Parallel K-Nearest Neighbors (KNN) Algorithm

This project implements a parallel version of the K-Nearest Neighbors (KNN) algorithm using the Message Passing Interface (MPI) with `mpi4py`. The parallelization allows for efficient computation of predictions by distributing the workload across multiple processes.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The K-Nearest Neighbors algorithm is a simple, non-parametric, and lazy learning algorithm used for classification and regression. This project enhances the standard KNN by parallelizing the computation, making it suitable for large datasets and computational environments with multiple processors.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/s4bino/parallel-knn
   cd ./src
   ```

2. **Install the required Python packages:**

   Make sure you have Python installed. You can install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include:

   ```text
   numpy
   mpi4py
   ```

3. **Set up MPI:**

   Ensure that MPI is installed on your system. You can install it using:

   - On Ubuntu:

     ```bash
     sudo apt-get install mpich
     ```

   - On MacOS:

     ```bash
     brew install mpich
     ```

## Usage

To run the KNN algorithm in parallel:

1. **Prepare your dataset:**

   Load your training data (`X_train`, `y_train`) and test data (`X_test`) into numpy arrays.

2. **Run the KNN algorithm:**

   Use the following command to execute the script with MPI, where `n` is the number of processes:

   ```bash
   mpiexec -n <number_of_processes> python knn_parallel.py
   ```

3. **Example:**

   Here is an example of how you might use the `KNearestNeighbors` class in your script:

   ```python
   from knn_parallel import KNearestNeighbors
   import numpy as np

   # Example dataset
   X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
   y_train = np.array([0, 0, 1, 1])
   X_test = np.array([[1, 2], [4, 5]])

   knn = KNearestNeighbors(k=3)
   knn.fit(X_train, y_train)
   predictions = knn.predict(X_test)

   if predictions is not None:  # Only rank 0 will have predictions
       print(predictions)
   ```

## Code Overview

- **`knn_parallel.py`:** Contains the `KNearestNeighbors` class, which implements the KNN algorithm with MPI parallelization.
- **`requirements.txt`:** Lists the Python dependencies for the project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
