#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h>  // For SIMD intrinsics (AVX)

#include "proj2-structs.h"


// Function for dense-dense matrix multiplication with multithreading, SIMD, and cache optimizations
Matrix matrix_multiply_dense_combined(const Matrix &A, const Matrix &B) {
    if (A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    int result_rows = A.size();
    int result_cols = B[0].size();
    int inner_dim = B.size();
    Matrix result(result_rows, std::vector<double>(result_cols, 0.0));

    // Timer start
    auto start_time = std::chrono::high_resolution_clock::now();

    // Parallelize outer loop over rows of matrix A
    #pragma omp parallel for
    for (int i = 0; i < result_rows; ++i) {
        // Loop over columns of matrix B
        for (int j = 0; j < result_cols; ++j) {
            // Compute dot product of the ith row of A and the jth column of B
            __m256d sum_vec = _mm256_setzero_pd();  // SIMD vector to accumulate results

            // Loop through the common dimension (inner_dim) using SIMD
            for (int k = 0; k < inner_dim - (inner_dim % 4); k += 4) {
                // Load 4 elements from row of A and column of B
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]);
                __m256d b_vec = _mm256_set_pd(B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]);

                // Perform SIMD multiplication and accumulate the results
                sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
            }

            // Sum up the partial results in the vector
            double sum[4];
            _mm256_storeu_pd(sum, sum_vec);
            result[i][j] = sum[0] + sum[1] + sum[2] + sum[3];

            // Handle any remaining elements that weren't handled by SIMD
            for (int k = inner_dim - (inner_dim % 4); k < inner_dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Timer stop
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Time taken: " << duration.count() << " s\n";
    // std::cout.flush();

    return result;
}

// Function for dense-sparse matrix multiplication with multithreading, SIMD, and cache optimizations
Matrix matrix_multiply_dense_sparse_combined(const Matrix &dense, const SparseMatrix &sparse) {
    int result_rows = dense.size();
    int result_cols = sparse.cols;
    Matrix result(result_rows, std::vector<double>(result_cols, 0.0));

    // Parallelize across rows of the dense matrix
    #pragma omp parallel for
    for (int i = 0; i < result_rows; ++i) {
        const std::vector<double> &dense_row = dense[i];  // Row of the dense matrix

        // Iterate over rows of the sparse matrix
        for (int j = 0; j < sparse.rows; ++j) {
            // Get non-zero elements in row j of the sparse matrix
            for (int k = sparse.row_pointers[j]; k < sparse.row_pointers[j + 1]; ++k) {
                int col_idx = sparse.col_indices[k];  // Column index of non-zero element
                double sparse_value = sparse.values[k];

                // Accumulate the result without SIMD
                result[i][col_idx] += dense_row[j] * sparse_value;
            }
        }
    }

    return result;
}

// Function to transpose a sparse matrix
SparseMatrix transpose_sparse(const SparseMatrix &B) {
    SparseMatrix B_transposed;
    B_transposed.rows = B.cols;
    B_transposed.cols = B.rows;
    B_transposed.row_pointers.resize(B.cols + 1, 0);

    // Count non-zeros per column (which will become row in transposed)
    std::vector<int> counts(B.cols, 0);
    for (int i = 0; i < B.row_pointers.size() - 1; ++i) {
        for (int j = B.row_pointers[i]; j < B.row_pointers[i + 1]; ++j) {
            counts[B.col_indices[j]]++;
        }
    }

    // Build row_pointers for transposed
    for (int i = 1; i <= B.cols; ++i) {
        B_transposed.row_pointers[i] = B_transposed.row_pointers[i - 1] + counts[i - 1];
    }

    B_transposed.values.resize(B.values.size());
    B_transposed.col_indices.resize(B.col_indices.size());

    // Fill in values and col_indices for transposed matrix
    std::vector<int> current_index(B.cols, 0);
    for (int i = 0; i < B.rows; ++i) {
        for (int j = B.row_pointers[i]; j < B.row_pointers[i + 1]; ++j) {
            int colB = B.col_indices[j];
            int dest_idx = B_transposed.row_pointers[colB] + current_index[colB];

            B_transposed.values[dest_idx] = B.values[j];
            B_transposed.col_indices[dest_idx] = i;
            current_index[colB]++;
        }
    }

    return B_transposed;
}

// Function for sparse-sparse matrix multiplication with multithreading, SIMD, and cache optimizations
SparseMatrix matrix_multiply_sparse_sparse_combined(const SparseMatrix &A, const SparseMatrix &B, int blockSize) {
    SparseMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.row_pointers.push_back(0);  // Start of first row

    // Temporary local buffers for each thread
    std::vector<std::vector<double>> temp_values(A.rows);
    std::vector<std::vector<int>> temp_col_indices(A.rows);

    // Perform parallelized sparse-sparse matrix multiplication
    #pragma omp parallel for
    for (int ii = 0; ii < A.rows; ii += blockSize) {
        for (int row_a = ii; row_a < std::min(ii + blockSize, A.rows); ++row_a) {
            std::vector<double> row_result(B.cols, 0.0);  // Local row buffer

            for (int idx_a = A.row_pointers[row_a]; idx_a < A.row_pointers[row_a + 1]; ++idx_a) {
                int col_a = A.col_indices[idx_a];
                double val_a = A.values[idx_a];

                __m256d val_a_vec = _mm256_set1_pd(val_a);  // Broadcast A's non-zero value to all lanes

                // Multiply non-zero elements of A's row with B's row (where non-zero values match)
                int idx_b;
                for (idx_b = B.row_pointers[col_a]; idx_b + 4 <= B.row_pointers[col_a + 1]; idx_b += 4) {
                    int col_b = B.col_indices[idx_b];

                    // Load non-zero values from B
                    __m256d b_values = _mm256_loadu_pd(&B.values[idx_b]);
                    __m256d c_values = _mm256_loadu_pd(&row_result[col_b]);

                    // Perform vectorized multiply-add (C += A * B)
                    c_values = _mm256_fmadd_pd(val_a_vec, b_values, c_values);

                    // Store the result back into the local result buffer
                    _mm256_storeu_pd(&row_result[col_b], c_values);
                }

                // Handle leftover elements not divisible by 4
                for (; idx_b < B.row_pointers[col_a + 1]; ++idx_b) {
                    int col_b = B.col_indices[idx_b];
                    row_result[col_b] += val_a * B.values[idx_b];
                }
            }

            // Store the non-zero results into the thread-local storage
            for (int j = 0; j < B.cols; ++j) {
                if (row_result[j] != 0.0) {
                    #pragma omp critical
                    {
                        temp_values[row_a].push_back(row_result[j]);
                        temp_col_indices[row_a].push_back(j);
                    }
                }
            }
        }
    }

    // Construct the final sparse matrix C from the temporary results
    for (int row = 0; row < A.rows; ++row) {
        C.values.insert(C.values.end(), temp_values[row].begin(), temp_values[row].end());
        C.col_indices.insert(C.col_indices.end(), temp_col_indices[row].begin(), temp_col_indices[row].end());
        C.row_pointers.push_back(C.values.size());
    }

    return C;
}

// Function to test dense-dense matrix multiplication with multi-threading
void test_dense_combined_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    Matrix B_dense = initialize_dense_matrix(N, N);

    
    Matrix C_dense = matrix_multiply_dense_combined(A_dense, B_dense);
}

// Function to test dense-sparse matrix multiplication with multi-threading
void test_dense_sparse_combined_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_dense_sparse = matrix_multiply_dense_sparse_combined(A_dense, B_sparse);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " s\n";
}

// Function to test sparse-sparse matrix multiplication with multi-threading
void test_sparse_sparse_combined_multiplication(int N, double sparsity, int blockSize) {
    
    std::cout << N << "\t";

    // Initialize matrices
    SparseMatrix A_sparse = initialize_sparse_matrix(N, N, sparsity);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);

    auto start = std::chrono::high_resolution_clock::now();
    SparseMatrix C_sparse = matrix_multiply_sparse_sparse_combined(A_sparse, B_sparse, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " s\n";
}

int main() {
    // Test with matrix size N and different sparsities (fastest first)
    int blockSize = 64;

    // Case 3: Sparse-Sparse Multiplication
    std::cout << "Case 3: Sparse-Sparse Multiplication\n\n";
    double sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_combined_multiplication(N, sparsity, blockSize);
    }
    
    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_combined_multiplication(N, sparsity, blockSize);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_combined_multiplication(N, sparsity, blockSize);
    }

    // Case 2: Dense-Sparse Multiplication
    std::cout << "\nCase 2: Dense-Sparse Multiplication\n\n";
    sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_combined_multiplication(N, sparsity);
    }

    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_combined_multiplication(N, sparsity);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_combined_multiplication(N, sparsity);
    }

    // Case 1: Dense-Dense Multiplication
    std::cout << "\nCase 1: Dense-Dense Multiplication\n\n";
    sparsity = 0.1;
    for (int i = 1; i < 11; ++i) {
        int N = i * 200;
        test_dense_combined_multiplication(N, sparsity);
    }

    return 0;
}


