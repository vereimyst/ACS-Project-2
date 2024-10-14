#include <iostream>
#include <chrono>
#include <cstdlib>

#include "proj2-structs.h"


// Naive dense-dense matrix multiplication
Matrix matrix_multiply_dense(const Matrix &A, const Matrix &B) {
    int n = A.size();      // Rows in A
    int m = B.size();      // Rows in B
    int p = B[0].size();   // Columns in B

    // Initialize result matrix C with zeros
    Matrix C(n, std::vector<double>(p, 0.0));

    // Perform naive matrix multiplication (C = A * B)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < m; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Dense-sparse matrix multiplication
Matrix matrix_multiply_dense_sparse(const Matrix &A, const SparseMatrix &B) {
    int n = A.size();      // Rows in A
    int m = A[0].size();   // Columns in A (must equal rows in B)
    int p = B.cols;        // Columns in B

    // Initialize result matrix C with zeros
    Matrix C(n, std::vector<double>(p, 0.0));

    // Multiply dense matrix A by sparse matrix B
    for (int i = 0; i < n; ++i) {
        for (int row_b = 0; row_b < B.rows; ++row_b) {
            for (int idx = B.row_pointers[row_b]; idx < B.row_pointers[row_b + 1]; ++idx) {
                int col_b = B.col_indices[idx];
                C[i][col_b] += A[i][row_b] * B.values[idx];
            }
        }
    }

    return C;
}

// Sparse-sparse matrix multiplication
SparseMatrix matrix_multiply_sparse_sparse(const SparseMatrix &A, const SparseMatrix &B) {
    // We'll use a simplified method that multiplies two sparse matrices
    SparseMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.row_pointers.push_back(0); // Start of first row

    for (int row_a = 0; row_a < A.rows; ++row_a) {
        std::vector<double> row_result(B.cols, 0.0);

        // Multiply row_a of A with columns of B
        for (int idx_a = A.row_pointers[row_a]; idx_a < A.row_pointers[row_a + 1]; ++idx_a) {
            int col_a = A.col_indices[idx_a];
            double val_a = A.values[idx_a];

            for (int idx_b = B.row_pointers[col_a]; idx_b < B.row_pointers[col_a + 1]; ++idx_b) {
                int col_b = B.col_indices[idx_b];
                row_result[col_b] += val_a * B.values[idx_b];
            }
        }

        // Store the result row in sparse format
        for (int j = 0; j < B.cols; ++j) {
            if (row_result[j] != 0.0) {
                C.values.push_back(row_result[j]);
                C.col_indices.push_back(j);
            }
        }
        C.row_pointers.push_back(C.values.size());
    }

    return C;
}

void test_dense_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    Matrix B_dense = initialize_dense_matrix(N, N);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_dense = matrix_multiply_dense(A_dense, B_dense);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " ms\n";
}

void test_dense_sparse_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_dense_sparse = matrix_multiply_dense_sparse(A_dense, B_sparse);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " ms\n";
}

void test_sparse_sparse_multiplication(int N, double sparsity) {
    std::cout << N << "\t";

    // Initialize matrices
    SparseMatrix A_sparse = initialize_sparse_matrix(N, N, sparsity);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);

    auto start = std::chrono::high_resolution_clock::now();
    SparseMatrix C_sparse = matrix_multiply_sparse_sparse(A_sparse, B_sparse);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " ms\n";
}

void test_multiplication(int N, double sparsity) {
    std::cout << "--- " << N << "x" << N << " ---\n";

    // Initialize matrices
    Matrix A_dense = initialize_dense_matrix(N, N);
    Matrix B_dense = initialize_dense_matrix(N, N);
    SparseMatrix A_sparse = initialize_sparse_matrix(N, N, sparsity);
    SparseMatrix B_sparse = initialize_sparse_matrix(N, N, sparsity);

    // Case 1: Dense-Dense Multiplication
    std::cout << "Case 1: Dense-Dense Multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_dense = matrix_multiply_dense(A_dense, B_dense);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n\n";

    // Case 2: Dense-Sparse Multiplication
    std::cout << "Case 2: Dense-Sparse Multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    Matrix C_dense_sparse = matrix_multiply_dense_sparse(A_dense, B_sparse);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n\n";

    // Case 3: Sparse-Sparse Multiplication
    std::cout << "Case 3: Sparse-Sparse Multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    SparseMatrix C_sparse = matrix_multiply_sparse_sparse(A_sparse, B_sparse);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n\n";
}

int main() {
    // Test with matrix size N and different sparsities
    // double sparsity = 0.1;
    // std::cout << "---------- Sparsity 10% ----------\n";
    // for (int i = 1; i < 4; ++i) {
    //     int N = i * 1000;
    //     test_multiplication(N, sparsity);
    // }

    // sparsity = 0.01;
    // std::cout << "---------- Sparsity 1% ----------\n";
    // for (int i = 1; i < 4; ++i) {
    //     int N = i * 1000;
    //     test_multiplication(N, sparsity);
    // }

    // sparsity = 0.001;
    // std::cout << "---------- Sparsity 0.1% ----------\n";
    // for (int i = 3; i < 4; ++i) {
    //     int N = i * 1000;
    //     test_multiplication(N, sparsity);
    // }

    // Case 3: Sparse-Sparse Multiplication
    std::cout << "Case 3: Sparse-Sparse Multiplication\n\n";
    double sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_multiplication(N, sparsity);
    }
    
    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_multiplication(N, sparsity);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 11; ++i) {
        int N = i * 1000;
        test_sparse_sparse_multiplication(N, sparsity);
    }

    // Case 2: Dense-Sparse Multiplication
    std::cout << "\nCase 2: Dense-Sparse Multiplication\n\n";
    sparsity = 0.1;
    std::cout << "---------- Sparsity 10% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_multiplication(N, sparsity);
    }

    sparsity = 0.01;
    std::cout << "---------- Sparsity 1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_multiplication(N, sparsity);
    }

    sparsity = 0.001;
    std::cout << "---------- Sparsity 0.1% ----------\n";
    for (int i = 1; i < 7; ++i) {
        int N = i * 500;
        test_dense_sparse_multiplication(N, sparsity);
    }

    // Case 1: Dense-Dense Multiplication
    std::cout << "\nCase 1: Dense-Dense Multiplication\n\n";
    sparsity = 0.1;
    for (int i = 1; i < 11; ++i) {
        int N = i * 200;
        test_dense_multiplication(N, sparsity);
    }

    return 0;
}
