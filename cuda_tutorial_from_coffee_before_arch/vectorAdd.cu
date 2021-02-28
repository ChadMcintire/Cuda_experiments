#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

//CUDA kernel for vector addition
// __global__ means this called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {

    //Calculate global thread ID of 1D thread
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    //do a Boundary check to make sure we are not getting memory we don't own
    if(tid < N)
        c[tid] = a[tid] + b[tid];
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &
c) 
{
    for (int i = 0; i < a.size(); i++) 
    {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    //bitwise left shift, comes out to 2^16
    constexpr int N = 1 << 16;
    std::cout << "value of N = " << N;
    constexpr size_t bytes = sizeof(int) * N;

    //preallocate size at compile time so we don't get a runtime
    //performance hit
    std::vector<int> a;
    a.reserve(N);

    std::vector<int> b;
    b.reserve(N);

    std::vector<int> c;
    c.reserve(N);

    // Initialize random numbers in each array
    // between 0 to 100
    for (int i = 0; i < N; i++) {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data from the host to the device (CPU -> GPU)
    // synchronous call
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    //threads per CTA (1024)
    int NUM_THREADS = 1 << 10;

    //CTA's per Grid
    //We launch at LEAST as many threads as we have elements
    //This equation pads an extra CTA to the grid if N cannot evenly be 
    // divided by NUM_THREADS (eg. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    std::cout << "\nvalue of NUM_BLOCKS = " << NUM_BLOCKS;

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    //Copy sum vector from device to host
    //cudaMemcpy is a synchronous operation, and waits for the prior kernel
    //launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier
    // syncronous call
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors, this is a cpu check 
    verify_result(a, b, c);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    std::cout << "\nVector addition complete";

    return 0;
}
