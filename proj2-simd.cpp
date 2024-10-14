#include <iostream>
#include <chrono>
#include <immintrin.h>  // AVX Intrinsics

#include "proj2-structs.h"

// SIMD-optimized matrix-matrix multiplication using AVX
Matrix matrix_multiply_dense_simd(const Matrix &A, const Matrix &B) {
    int m = A.size();        // Number of rows in matrix A
    int k = A[0].size();     // Number of columns in matrix A (and rows in matrix B)
    int n = B[0].size();     // Number of columns in matrix B

    // Initialize result matrix C (dense) with zeros
    Matrix C(m, std::vector<double>(n, 0.0));

    // Start timing the matrix multiplication
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication A * B = C using SIMD optimization
    for (int i = 0; i < m; ++i) {                // Iterate over rows of A
        for (int j = 0; j < n; ++j) {            // Iterate over columns of B
            __m256d c_vec = _mm256_setzero_pd(); // Initialize vector to accumulate result for C[i][j]

            for (int p = 0; p < k; p += 4) {     // Process 4 elements of A and B at a time
                // Load 4 elements from A[i][p:p+4] into a SIMD register
                __m256d a_vec = _mm256_loadu_pd(&A[i][p]);

                // Load 4 elements from B[p:p+4][j] into a SIMD register
                __m256d b_vec = _mm256_loadu_pd(&B[p][j]);

                // Perform fused multiply-add: C[i][j] += A[i][p:p+4] * B[p:p+4][j]
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }

            // Accumulate the result from the SIMD register into the scalar result
            C[i][j] = c_vec[0] + c_vec[1] + c_vec[2] + c_vec[3];
        }
    }

    // Stop timing the matrix multiplication
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Time taken: " << elapsed.count() << " s\n";

    return C;
}

// SIMD-optimized dense-sparse matrix multiplication using AVX
Matrix matrix_multiply_dense_sparse_simd(const Matrix &A, const SparseMatrix &B) {
    int m = A.size();        // Number of rows in dense matrix A
    int k = A[0].size();     // Number of columns in A (should match rows in B)
    int n = B.cols;          // Number of columns in sparse matrix B

    // Initialize result matrix C (dense) with zeros
    Matrix C(m, std::vector<double>(n, 0.0));

    // Start timing the matrix multiplication
    auto start_time = std::chrono::high_resolution_clock::now();

    // Multiply dense matrix A by sparse matrix B in CSR format
    for (int i = 0; i < m; ++i) {  // Iterate over rows of A
        for (int row_B = 0; row_B < B.rows; ++row_B) {
            double val_A = A[i][row_B];

            // Only process if the element in A is not zero (in practice, most are not zero)
            if (val_A != 0.0) {
                for (int idx = B.row_pointers[row_B]; idx < B.row_pointers[row_B + 1]; ++idx) {
                    int col_B = B.col_indices[idx];
                    double val_B = B.values[idx];

                    // Perform SIMD multiplication
                    __m256d a_vec = _mm256_set1_pd(val_A);
                    __m256d b_vec = _mm256_set1_pd(val_B);
                    __m256d c_vec = _mm256_set1_pd(C[i][col_B]);

                    // Fused multiply-add: C[i][col_B] += A[i][row_B] * B[row_B][col_B]
                    __m256d result = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                    // Store the result back into C[i][col_B]
                    C[i][col_B] += result[0];  // Store only the first element (as we're dealing with single values)
                }
            }
        }
    }

    // Stop timing the matrix multiplication
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Time taken: " << elapsed.count() << " s\n";

    return C;
}

// SIMD-optimized sparse-sparse matrix multiplication (partial)
SparseMatrix matrix_multiply_sparse_sparse_simd(const SparseMatrix &A, const SparseMatrix &B) {
    SparseMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.row_pointers.push_back(0);  // Start row pointers for C

    // Temporary storage for the non-zero values and column indices of C
    std::vector<double> temp_values;
    std::vector<int> temp_col_indices;

    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform sparse-sparse matrix multiplication
    for (int i = 0; i < A.rows; ++i) {
        std::vector<double> row_C(B.cols, 0.0);  // Temporary result for row i of C

        for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; ++j) {
            int col_A = A.col_indices[j];
            double val_A = A.values[j];

            // Multiply row i of A by the corresponding columns in B
            for (int k = B.row_pointers[col_A]; k < B.row_pointers[col_A + 1]; ++k) {
                int col_B = B.col_indices[k];
                double val_B = B.values[k];

                // Perform multiplication and accumulate the result
                __m256d a_vec = _mm256_set1_pd(val_A);
                __m256d b_vec = _mm256_set1_pd(val_B);
                __m256d c_vec = _mm256_set1_pd(row_C[col_B]);

                __m256d result = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                row_C[col_B] += result[0];  // Store the result (assuming the first value for simplicity)
            }
        }

        // Store non-zero results from row_C into C's sparse format
        for (int col = 0; col < B.cols; ++col) {
            if (row_C[col] != 0.0) {
                temp_values.push_back(row_C[col]);
                temp_col_indices.push_back(col);
            }
        }

        // Update row pointer for C
        C.row_pointers.push_back(temp_values.size());
    }

    // Populate the values and column indices of C
    C.values = temp_values;
    C.col_indices = temp_col_indices;

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Time taken: " << elapsed.count() << " s\n";

    return C;
}

// Function to test dense-dense matrix multiplication with multi-threading
void test_dense_simd_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    Matrix B_dense = initialize_dense_matrix(N, N);

    Matrix C_dense = matrix_multiply_dense_simd(A_dense, B_dense);
}

// Function to test dense-sparse matrix multiplication with multi-threading
void test_dense_sparse_simd_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);
    
    Matrix C_dense_sparse = matrix_multiply_dense_sparse_simd(A_dense, B_sparse);
}

// Function to test sparse-sparse matrix multiplication with multi-threading
void test_sparse_sparse_simd_multiplication(int N, double sparsity) {
    
    std::cout << N << "\t";

    // Initialize matrices
    SparseMatrix A_sparse = initialize_sparse_matrix(N, N, sparsity);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);

    SparseMatrix C_sparse = matrix_multiply_sparse_sparse_simd(A_sparse, B_sparse);
}

int main() {
    // Test with matrix size N and different sparsities (fastest first)

    // // Case 3: Sparse-Sparse Multiplication
    // std::cout << "Case 3: Sparse-Sparse Multiplication\n\n";
    double sparsity = 0.1;
    // std::cout << "---------- Sparsity 10% ----------\n";
    // for (int i = 1; i < 11; ++i) {
    //     int N = i * 1000;
    //     test_sparse_sparse_simd_multiplication(N, sparsity);
    // }
    
    // sparsity = 0.01;
    // std::cout << "---------- Sparsity 1% ----------\n";
    // for (int i = 1; i < 11; ++i) {
    //     int N = i * 1000;
    //     test_sparse_sparse_simd_multiplication(N, sparsity);
    // }

    // sparsity = 0.001;
    // std::cout << "---------- Sparsity 0.1% ----------\n";
    // for (int i = 1; i < 11; ++i) {
    //     int N = i * 1000;
    //     test_sparse_sparse_simd_multiplication(N, sparsity);
    // }

    // Case 2: Dense-Sparse Multiplication
    std::cout << "\nCase 2: Dense-Sparse Multiplication\n\n";
    sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_simd_multiplication(N, sparsity);
    }

    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_simd_multiplication(N, sparsity);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_simd_multiplication(N, sparsity);
    }

    // Case 1: Dense-Dense Multiplication
    std::cout << "\nCase 1: Dense-Dense Multiplication\n\n";
    sparsity = 0.1;
    for (int i = 1; i < 11; ++i) {
        int N = i * 200;
        test_dense_simd_multiplication(N, sparsity);
    }

    return 0;
}
