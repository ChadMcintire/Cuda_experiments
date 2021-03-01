#include <iostream>
#include <cassert>

//CUDA kernel for vector addition
//No change when using CUDA unified memory
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    //calculate global thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    //Boundary check
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    //Array size of 2^16 (65536 elements), bitshift left
    const int N = 1 << 16;
    size_t bytes = N * sizeof(int);

    //Declare unified memory pointers
    int *a, *b, *c;

    //Allocating memory for these pointer
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    //Get the device ID for prefetching calls
    int id = cudaGetDevice(&id);

    std::cout << "Device ids " << id << '\n';

    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);

    //Initialize vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    //Pre-fetch 'a' and 'b' arrays to the specified device (GPU)
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    //Threads per CTA (1024 threads per CTA)
    int BLOCK_SIZE = 1 << 10;

    //CTA's per grid
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //Call CUDA kernel
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

    // wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization
    // of cudaMemcpy like in the origin example
    cudaDeviceSynchronize();

    //Prefetch to the host (CPU)
    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    //Verify the result on the CPU
    for (int i = 0; i < N; i++) {
        assert(c[i] == a[i] + b[i]);
    }

    //Free unified memory (same as memory allocated with cudaMalloc)
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

   std::cout << "Completed successfully!\n";

   return 0;

}
