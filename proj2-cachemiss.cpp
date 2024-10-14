#include <iostream>
#include <chrono>
#include <cstdlib>

#include "proj2-structs.h"


// Dense-Dense matrix multiplication using loop blocking (cache optimization)
Matrix matrix_multiply_dense_blocking(const Matrix &A, const Matrix &B, int blockSize) {
    int A_rows = A.size();
    int A_cols = A[0].size();
    int B_cols = B[0].size();

    if (A_cols != B.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Initialize result matrix C with zeros
    Matrix C(A_rows, std::vector<double>(B_cols, 0.0));

    // Loop over the blocks for better cache utilization
    for (int ii = 0; ii < A_rows; ii += blockSize) {
        for (int jj = 0; jj < B_cols; jj += blockSize) {
            for (int kk = 0; kk < A_cols; kk += blockSize) {
                // Multiply block A[ii:ii+blockSize, kk:kk+blockSize] by block B[kk:kk+blockSize, jj:jj+blockSize]
                for (int i = ii; i < std::min(ii + blockSize, A_rows); ++i) {
                    for (int j = jj; j < std::min(jj + blockSize, B_cols); ++j) {
                        double sum = 0.0;
                        for (int k = kk; k < std::min(kk + blockSize, A_cols); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }

    return C;
}

// Dense-sparse matrix multiplication using loop blocking (cache optimization)
Matrix matrix_multiply_dense_sparse_blocking(const Matrix &dense, const SparseMatrix &sparse) {
    int dense_rows = dense.size();
    int dense_cols = dense[0].size();
    int sparse_cols = sparse.cols;

    // Initialize the result matrix
    Matrix result(dense_rows, std::vector<double>(sparse_cols, 0.0));

    // Multiply dense matrix by sparse matrix
    for (int i = 0; i < dense_rows; ++i) {
        for (int row_a = 0; row_a < sparse.rows; ++row_a) {
            for (int idx_a = sparse.row_pointers[row_a]; idx_a < sparse.row_pointers[row_a + 1]; ++idx_a) {
                int col_a = sparse.col_indices[idx_a];
                double val_a = sparse.values[idx_a];

                // Perform multiplication and accumulate the result in the corresponding row and column
                result[i][col_a] += dense[i][row_a] * val_a;
            }
        }
    }

    return result;
}

// Sparse-sparse matrix multiplication using loop blocking (cache optimization)
SparseMatrix matrix_multiply_sparse_sparse_blocking(const SparseMatrix &A, const SparseMatrix &B) {
    SparseMatrix result;
    result.rows = A.rows;
    result.cols = B.cols;
    result.row_pointers.push_back(0);  // First row starts at index 0

    std::vector<double> row_result(B.cols, 0.0);
    std::vector<int> col_flags(B.cols, -1); // To track non-zero columns

    for (int i = 0; i < A.rows; ++i) {
        int row_start_A = A.row_pointers[i];
        int row_end_A = A.row_pointers[i + 1];

        for (int j = row_start_A; j < row_end_A; ++j) {
            int col_A = A.col_indices[j];
            double value_A = A.values[j];

            int row_start_B = B.row_pointers[col_A];
            int row_end_B = B.row_pointers[col_A + 1];

            for (int k = row_start_B; k < row_end_B; ++k) {
                int col_B = B.col_indices[k];
                double value_B = B.values[k];

                if (col_flags[col_B] == -1) {
                    col_flags[col_B] = result.values.size();
                    result.values.push_back(value_A * value_B);
                    result.col_indices.push_back(col_B);
                } else {
                    result.values[col_flags[col_B]] += value_A * value_B;
                }
            }
        }

        result.row_pointers.push_back(result.values.size());

        // Reset the col_flags and row_result for the next row
        for (int j = result.row_pointers[i]; j < result.row_pointers[i + 1]; ++j) {
            col_flags[result.col_indices[j]] = -1;
        }
    }

    return result;
}

// Function to test dense-dense matrix multiplication with multi-threading
void test_dense_blocking_multiplication(int N, double sparsity, int blockSize) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    Matrix B_dense = initialize_dense_matrix(N, N);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_dense = matrix_multiply_dense_blocking(A_dense, B_dense, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " s\n";
}

// Function to test dense-sparse matrix multiplication with multi-threading
void test_dense_sparse_blocking_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_dense_sparse = matrix_multiply_dense_sparse_blocking(A_dense, B_sparse);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " s\n";
}

// Function to test sparse-sparse matrix multiplication with multi-threading
void test_sparse_sparse_blocking_multiplication(int N, double sparsity) {
    
    std::cout << N << "\t";

    // Initialize matrices
    SparseMatrix A_sparse = initialize_sparse_matrix(N, N, sparsity);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);

    auto start = std::chrono::high_resolution_clock::now();
    SparseMatrix C_sparse = matrix_multiply_sparse_sparse_blocking(A_sparse, B_sparse);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " s\n";
}

int main() {
    // Test with matrix size N and different sparsities (fastest first)
    int block_size = 64;

    // Case 3: Sparse-Sparse Multiplication
    std::cout << "Case 3: Sparse-Sparse Multiplication\n\n";
    double sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_blocking_multiplication(N, sparsity);
    }
    
    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_blocking_multiplication(N, sparsity);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_blocking_multiplication(N, sparsity);
    }

    // Case 2: Dense-Sparse Multiplication
    std::cout << "\nCase 2: Dense-Sparse Multiplication\n\n";
    sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_blocking_multiplication(N, sparsity);
    }

    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_blocking_multiplication(N, sparsity);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_blocking_multiplication(N, sparsity);
    }

    // Case 1: Dense-Dense Multiplication
    std::cout << "\nCase 1: Dense-Dense Multiplication\n\n";
    sparsity = 0.1;
    for (int i = 1; i < 11; ++i) {
        int N = i * 200;
        test_dense_blocking_multiplication(N, sparsity, block_size);
    }

    return 0;
}