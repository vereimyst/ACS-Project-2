# ACS-Project-2

This project implements, in each respective file, a method of matrix multiplication. All methods share the same representation for Sparse Matrices, Compressed Sparse Row (CSR) Format.
The naive implementaion is compiled using `g++ proj2-naive.cpp -o naive` and available to run directly via `./naive`.

Multithreading uses the built-in C++ thread library and an explicit thread count of 16, as is available on my laptop. Running on a different local environment may require a different value to be set there (line 212). I compiled using `g++ proj2-multithread.cpp -o multithread` and it can be run using `./multithread`.

Due to the specific hardware requirements for SIMD to function, I used `g++ -mavx2 -mfma -o simd proj2-simd.cpp`. However, this may vary depending on your setup. Assuming your CPU has the same specs as mine, you can run the implementation directly with `./simd`.

Cache miss optimization is implemented using loop blocking. This was simply compiled with `g++ proj2-cachemiss.cpp -o cachemiss`. However, I used a block size of 64B, which is accommdated by my system. This value may be changed in line 161 of the file. Otherwise, this file can be run directly with `./cachemiss`.

My implementation combining the 3 above uses OpenMP for flexible threading dependent on the available threads on your device. Again because of my laptop's cache size and hardware setup, I compiled using `g++ -fopenmp -O3 -mavx -std=c++11 proj2-combined.cpp -o combined`. It can be run directly using `./combined`.