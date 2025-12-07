#include "../gpu-error.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/time.h>

using namespace std;
using namespace std::chrono;

template <typename T> __global__ void initKernel(T *data, size_t data_len) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = tidx; idx < data_len; idx += gridDim.x * blockDim.x) {
    data[idx] = idx;
  }
}

// Runtime version of testfun where N is passed as a parameter
template <typename T, int M, int BLOCKSIZE>
__global__ void testfun_runtime(T *const __restrict__ dA, T *const __restrict__ dB,
                                T *dC, int N) {
  T *sA = dA + threadIdx.x + blockIdx.x * BLOCKSIZE * M;
  T *sB = dB + threadIdx.x + blockIdx.x * BLOCKSIZE * M;

  T sum = 0;

  for (int i = 0; i < M; i += 2) {
    T a = sA[i * BLOCKSIZE];
    T b = sB[i * BLOCKSIZE];
    T v = a - b;
    T a2 = sA[(i + 1) * BLOCKSIZE];
    T b2 = sB[(i + 1) * BLOCKSIZE];
    T v2 = a2 - b2;
    for (int j = 0; j < N; j++) {
      v = v * a - b;
      v2 = v2 * a - b;
    }
    sum += v + v2;
  }
  if (threadIdx.x == 0)
    dC[blockIdx.x] = sum;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }

  double target_ai = atof(argv[1]);
  double duration_sec = atof(argv[2]);

  if (target_ai <= 0 || duration_sec <= 0) {
    return 1;
  }

  typedef float dtype;
  const int M = 4000;
  const int BLOCKSIZE = 256;
  
  // Sampling interval for TFLOPS measurement (default=10)
  const int SAMPLE_INTERVAL = 1000;

  // Calculate N from algorithmic intensity
  // AI = (2.0 + N * 2.0) / (2.0 * sizeof(dtype))
  // AI = (2.0 + N * 2.0) / 8.0
  // AI = (1.0 + N) / 4.0
  // 4 * AI = 1.0 + N
  // N = 4 * AI - 1
  int N = (int)round(4.0 * target_ai - 1.0);
  if (N < 0) N = 0;

  int nDevices;
  GPU_ERROR(cudaGetDeviceCount(&nDevices));

#pragma omp parallel num_threads(nDevices)
  {
    GPU_ERROR(cudaSetDevice(omp_get_thread_num()));
#pragma omp barrier
    int deviceId;
    GPU_ERROR(cudaGetDevice(&deviceId));
    cudaDeviceProp prop;
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    int numBlocks;

    // Use reasonable default for block count
    numBlocks = 8;
    int blockCount = prop.multiProcessorCount * numBlocks;

    size_t data_len = (size_t)blockCount * BLOCKSIZE * M;
    dtype *dA = NULL;
    dtype *dB = NULL;
    dtype *dC = NULL;

    GPU_ERROR(cudaMalloc(&dA, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dB, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dC, data_len * sizeof(dtype)));
#pragma omp barrier
    initKernel<<<blockCount, 256>>>(dA, data_len);
    initKernel<<<blockCount, 256>>>(dB, data_len);
    initKernel<<<blockCount, 256>>>(dC, data_len);
    GPU_ERROR(cudaDeviceSynchronize());

#pragma omp barrier

    auto start_time = steady_clock::now();
    auto end_time = start_time + duration<double>(duration_sec);

    // Calculate operations per kernel launch
    // Each kernel does: (2 + N * 2) operations per element
    size_t operations_per_kernel = (2 + N * 2) * data_len;

    // CUDA events for timing sampled kernels
    cudaEvent_t sample_start, sample_stop;
    GPU_ERROR(cudaEventCreate(&sample_start));
    GPU_ERROR(cudaEventCreate(&sample_stop));

    size_t iter_count = 0;

    // Run kernels continuously until time expires
    while (steady_clock::now() < end_time) {
      // Sample TFLOPS every SAMPLE_INTERVAL iterations
      bool should_sample = (iter_count % SAMPLE_INTERVAL == 0);
      
      if (should_sample) {
        GPU_ERROR(cudaEventRecord(sample_start));
      }
      
      testfun_runtime<dtype, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC, N);
      
      if (should_sample) {
        GPU_ERROR(cudaEventRecord(sample_stop));
        GPU_ERROR(cudaEventSynchronize(sample_stop));
        
        // Get timestamp in microseconds since epoch
        struct timeval tv;
        gettimeofday(&tv, NULL);
        long long timestamp_us = (long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec;
        
        // Calculate elapsed time for this kernel
        float elapsed_ms;
        GPU_ERROR(cudaEventElapsedTime(&elapsed_ms, sample_start, sample_stop));
        double elapsed_sec = elapsed_ms / 1000.0;
        
        // Calculate TFLOPS
        double tflops = (operations_per_kernel / elapsed_sec) / 1.0e12;
        
        // Log: timestamp_us,tflops
        // INSERT_YOUR_CODE
        
        std::ostringstream oss;
        oss << "gpu" << deviceId << "," << timestamp_us << "," << fixed << setprecision(6) << tflops <<"\n";
        cout << oss.str();
      }
      
      iter_count++;
    }

    GPU_ERROR(cudaEventDestroy(sample_start));
    GPU_ERROR(cudaEventDestroy(sample_stop));
    GPU_ERROR(cudaDeviceSynchronize());
    GPU_ERROR(cudaGetLastError());
    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
    GPU_ERROR(cudaFree(dC));
  }
  return 0;
}
