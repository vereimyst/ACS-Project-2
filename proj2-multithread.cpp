#include <iostream>
#include <chrono>
#include <omp.h>  // OpenMP for multi-threading
#include <thread>

#include "proj2-structs.h"


// Naive dense-dense matrix multiplication with OpenMP for multi-threading
Matrix matrix_multiply_dense_parallel(const Matrix &A, const Matrix &B, int num_threads) {
    int A_rows = A.size();
    int A_cols = A[0].size();
    int B_rows = B.size();
    int B_cols = B[0].size();

    // Ensure the dimensions are compatible for multiplication
    if (A_cols != B_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    // Initialize result matrix with 0s
    Matrix result(A_rows, std::vector<double>(B_cols, 0.0));

    // Helper function for multithreaded multiplication of a range of rows
    auto multiply_row_range = [&](int start_row, int end_row) {
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < B_cols; ++j) {
                for (int k = 0; k < A_cols; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    };

    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create threads
    std::vector<std::thread> threads;
    int rows_per_thread = A_rows / num_threads;

    // Divide work between threads
    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? A_rows : start_row + rows_per_thread;

        threads.emplace_back(multiply_row_range, start_row, end_row);
    }

    // Join all threads
    for (auto &thread : threads) {
        thread.join();
    }

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Time taken: " << duration << " ms\n";

    return result;
}

// Parallelized dense-sparse matrix multiplication using OpenMP
Matrix matrix_multiply_dense_sparse_parallel(const Matrix &dense, const SparseMatrix &sparse, int num_threads) {
    int dense_rows = dense.size();
    int dense_cols = dense[0].size();
    Matrix result(dense_rows, std::vector<double>(sparse.cols, 0.0));

    auto multiply_row_range = [&](int start_row, int end_row) {
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < sparse.rows; ++j) {
                double dense_val = dense[i][j];
                if (dense_val != 0) {
                    for (int idx = sparse.row_pointers[j]; idx < sparse.row_pointers[j + 1]; ++idx) {
                        int col = sparse.col_indices[idx];
                        result[i][col] += dense_val * sparse.values[idx];
                    }
                }
            }
        }
    };

    auto start_time = std::chrono::high_resolution_clock::now();

    // Multithreading setup
    std::vector<std::thread> threads;
    int rows_per_thread = dense_rows / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? dense_rows : start_row + rows_per_thread;

        threads.emplace_back(multiply_row_range, start_row, end_row);
    }

    // Join all threads
    for (auto &thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Time taken: " << duration << " ms\n";

    return result;
}

// Sparse-sparse matrix multiplication with OpenMP for multi-threading
SparseMatrix matrix_multiply_sparse_sparse_parallel(const SparseMatrix &A, const SparseMatrix &B, int num_threads) {
    SparseMatrix result;
    result.rows = A.rows;
    result.cols = B.cols;
    result.row_pointers.push_back(0);

    std::vector<std::vector<double>> row_values(A.rows);
    std::vector<std::vector<int>> row_col_indices(A.rows);

    auto multiply_row = [&](int row) {
        std::vector<double> temp_result(B.cols, 0.0);
        for (int idx_A = A.row_pointers[row]; idx_A < A.row_pointers[row + 1]; ++idx_A) {
            int col_A = A.col_indices[idx_A];
            double val_A = A.values[idx_A];

            for (int idx_B = B.row_pointers[col_A]; idx_B < B.row_pointers[col_A + 1]; ++idx_B) {
                int col_B = B.col_indices[idx_B];
                double val_B = B.values[idx_B];

                temp_result[col_B] += val_A * val_B;
            }
        }

        for (int col = 0; col < B.cols; ++col) {
            if (temp_result[col] != 0) {
                row_values[row].push_back(temp_result[col]);
                row_col_indices[row].push_back(col);
            }
        }
    };

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create threads to handle row-wise multiplication
    std::vector<std::thread> threads;
    int rows_per_thread = A.rows / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? A.rows : start_row + rows_per_thread;

        threads.emplace_back([&, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                multiply_row(i);
            }
        });
    }

    // Join threads
    for (auto &thread : threads) {
        thread.join();
    }

    // Collect results
    for (int i = 0; i < A.rows; ++i) {
        result.values.insert(result.values.end(), row_values[i].begin(), row_values[i].end());
        result.col_indices.insert(result.col_indices.end(), row_col_indices[i].begin(), row_col_indices[i].end());
        result.row_pointers.push_back(result.values.size());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Time taken: " << duration << " ms\n";

    return result;
}

// Function to test dense-dense matrix multiplication with multi-threading
void test_dense_parallel_multiplication(int N, double sparsity, int num_threads) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    Matrix B_dense = initialize_dense_matrix(N, N);

    Matrix C_dense = matrix_multiply_dense_parallel(A_dense, B_dense, num_threads);
}

// Function to test dense-sparse matrix multiplication with multi-threading
void test_dense_sparse_parallel_multiplication(int N, double sparsity, int num_threads) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);
    
    Matrix C_dense_sparse = matrix_multiply_dense_sparse_parallel(A_dense, B_sparse, num_threads);
}

// Function to test sparse-sparse matrix multiplication with multi-threading
void test_sparse_sparse_parallel_multiplication(int N, double sparsity, int num_threads) {
    
    std::cout << N << "\t";

    // Initialize matrices
    SparseMatrix A_sparse = initialize_sparse_matrix(N, N, sparsity);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);

    SparseMatrix C_sparse = matrix_multiply_sparse_sparse_parallel(A_sparse, B_sparse, num_threads);
}

int main() {
    // Test with matrix size N and different sparsities (fastest first)
    int num_threads = 16;

    // Case 3: Sparse-Sparse Multiplication
    std::cout << "Case 3: Sparse-Sparse Multiplication\n\n";
    double sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_parallel_multiplication(N, sparsity, num_threads);
    }
    
    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_parallel_multiplication(N, sparsity, num_threads);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_parallel_multiplication(N, sparsity, num_threads);
    }

    // Case 2: Dense-Sparse Multiplication
    std::cout << "\nCase 2: Dense-Sparse Multiplication\n\n";
    sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_parallel_multiplication(N, sparsity, num_threads);
    }

    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_parallel_multiplication(N, sparsity, num_threads);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_parallel_multiplication(N, sparsity, num_threads);
    }

    // Case 1: Dense-Dense Multiplication
    std::cout << "\nCase 1: Dense-Dense Multiplication\n\n";
    sparsity = 0.1;
    for (int i = 1; i < 11; ++i) {
        int N = i * 200;
        test_dense_parallel_multiplication(N, sparsity, num_threads);
    }

    return 0;
}
