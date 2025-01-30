# Randomized CUR Decomposition and Related Algorithms in Eigen

This repository contains an implementation of randomized and deterministic algorithms for the CUR decomposition, with a focus on Randomized CUR (RCUR). These algorithms utilize the Eigen C++ library for matrix operations.

## Overview of Algorithms

1. **Randomized SVD (RSVD)**: Used for dimensionality reduction or approximating the singular value decomposition of a matrix.
2. **Pseudoinverse**: Computes the Moore-Penrose pseudoinverse using Singular Value Decomposition (SVD).
3. **Subspace Iterations**: Iterative method to approximate dominant subspaces in matrices, often used in randomized SVD and CUR decomposition.
4. **Randomized QB Decomposition**: Used for dimensionality reduction by projecting the matrix to a lower-dimensional space.
5. **Deterministic CUR Decomposition (ID)**: A deterministic variant of CUR, where columns are selected based on pivoted QR decomposition.
6. **Randomized CUR Decomposition (RCUR)**: A randomized variant of CUR, combining techniques like subspace iteration, randomized QB decomposition, and pivoted QR decomposition.

## Key Functions

### 1. `generateRandomMatrix(int rows, int cols)`
Generates a random matrix with dimensions `rows` x `cols` with elements sampled from a Gaussian distribution (mean=0, standard deviation=1).

**Parameters**:
- `rows`: Number of rows in the matrix.
- `cols`: Number of columns in the matrix.

**Returns**:
- A matrix of random values.

### 2. `pseudoInverse(const MatrixXd& matrix)`
Computes the pseudoinverse of a given matrix using Singular Value Decomposition (SVD).

**Parameters**:
- `matrix`: The matrix for which the pseudoinverse is to be calculated.

**Returns**:
- The pseudoinverse of the input matrix.

### 3. `sub_iterations(const MatrixXd &A, MatrixXd Y, int q)`
Performs subspace iterations on matrix `Y` for `q` iterations to approximate a low-dimensional subspace of `A`.

**Parameters**:
- `A`: The input matrix.
- `Y`: The matrix to undergo subspace iterations.
- `q`: The number of iterations.

**Returns**:
- The resulting matrix after subspace iterations.

### 4. `rqb(const MatrixXd &A, int k, int p, int q)`
Performs Randomized QB Decomposition for dimensionality reduction, producing an orthonormal matrix `Q` and a low-dimensional matrix `B`.

**Parameters**:
- `A`: The input matrix.
- `k`: Target rank (dimension).
- `p`: Oversampling parameter for dimensionality reduction.
- `q`: Number of subspace iterations.

**Returns**:
- A pair of matrices `Q` and `B`.

### 5. `id(const MatrixXd &A, int k)`
Performs Deterministic Column ID (using pivoted QR decomposition) to extract low-rank approximations.

**Parameters**:
- `A`: The input matrix.
- `k`: Number of columns to extract.

**Returns**:
- A pair consisting of the expansion matrix `Z` and a vector `J` representing the selected columns.

### 6. `id_determ(const MatrixXd &A, int k)`
Performs Deterministic Column ID with QR decomposition and computes the expansion coefficients.

**Parameters**:
- `A`: The input matrix.
- `k`: The number of columns to extract.

**Returns**:
- A tuple consisting of the matrix `C`, the expansion matrix `Z`, and the selected column indices `J`.

### 7. `rid(const MatrixXd &A, int k, int p, int q)`
Performs Randomized ID (combining Randomized QB Decomposition and Deterministic Column ID).

**Parameters**:
- `A`: The input matrix.
- `k`: Target rank (dimension).
- `p`: Oversampling parameter for dimensionality reduction.
- `q`: Number of subspace iterations.

**Returns**:
- A tuple consisting of the matrix `C`, the expansion matrix `Z`, and the selected column indices `J`.

### 8. `rcur(const MatrixXd &A, int k, int p, int q, bool rand)`
Performs Randomized CUR Decomposition based on Randomized or Deterministic Column ID.

**Parameters**:
- `A`: The input matrix.
- `k`: Target rank (dimension).
- `p`: Oversampling parameter for dimensionality reduction.
- `q`: Number of subspace iterations.
- `rand`: Boolean flag to choose between randomized (`true`) or deterministic (`false`) column ID.

**Returns**:
- A tuple consisting of matrices `C`, `U`, and `R_mat` representing the decomposition.

## Usage

### Requirements:
- **Eigen C++ Library**: Ensure Eigen is installed in your environment, as this code relies on it for matrix operations.

### Example:
To use the `rcur` function for randomized CUR decomposition, you can include the necessary header and call the function as shown below:

```cpp
#include <Eigen/Dense>
#include "your_file.h" // Include the file containing the implementation

int main() {
    // Generate a random matrix A
    Eigen::MatrixXd A = generateRandomMatrix(100, 50);

    // Perform Randomized CUR Decomposition
    int k = 10; // Desired rank
    int p = 5;  // Oversampling
    int q = 2;  // Number of subspace iterations
    bool rand = true; // Use randomized ID

    auto [C, U, R] = rcur(A, k, p, q, rand);

    // Print the results
    std::cout << "C: \n" << C << std::endl;
    std::cout << "U: \n" << U << std::endl;
    std::cout << "R: \n" << R << std::endl;

    return 0;
}
```

### Compilation:
To compile the code with the Eigen library, use a C++ compiler like `g++`:

```bash
g++ -std=c++11 -I /path/to/eigen your_file.cpp -o your_program
./your_program
```

Replace `/path/to/eigen` with the path to the Eigen library on your system.

