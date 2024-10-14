#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>
#include <algorithm>    // For std::shuffle
#include <random>       // For std::random_device and std::mt19937

// Define matrix type using a 2D vector for dense matrices
typedef std::vector<std::vector<int>> Matrix;

// Sparse matrix in CSR format
struct SparseMatrix {
    std::vector<int> values;     // Non-zero values
    std::vector<int> col_indices;   // Column indices of the non-zero values
    std::vector<int> row_pointers;  // Row pointers (index of start of each row in 'values' array)
    int rows, cols;                 // Dimensions of the sparse matrix
};

// Function to initialize a dense matrix with random values
Matrix initialize_dense_matrix(int rows, int cols) {
    Matrix mat(rows, std::vector<int>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = rand() % 100;
        }
    }
    return mat;
}

// Function to initialize a sparse matrix with a given sparsity (percentage of non-zero elements)
SparseMatrix initialize_sparse_matrix(int rows, int cols, double sparsity) {
    SparseMatrix sparse;
    sparse.rows = rows;
    sparse.cols = cols;
    sparse.row_pointers.push_back(0); // Start of first row

    int sparse_size = rows * cols;
    int num_nonzeros = sparsity * sparse_size;
    std::vector<bool> nonzero(sparse_size);
    for (int i = 0; i < sparse_size; ++i) {
        if (i < num_nonzeros)   nonzero[i] = true;
        else                    nonzero[i] = false;
    }

    // Initialize a random number generator
    std::random_device rd;  // Seed for the random number generator
    std::mt19937 g(rd());   // Standard Mersenne Twister engine seeded with rd()

    // Shuffle the vector
    std::shuffle(nonzero.begin(), nonzero.end(), g);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (nonzero[i*j + j]) { // If non-zero element, create element
                sparse.values.push_back(rand() % 100); // Random non-zero value
                sparse.col_indices.push_back(j); // Column index of the non-zero element
            }
        }
        sparse.row_pointers.push_back(sparse.values.size()); // End of the row
    }

    return sparse;
}

#endif