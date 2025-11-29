#include "dtime.hpp"
#include "gpu-error.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <cmath>

#include "MeasurementSeries.hpp"

#include "gpu-stats.h"

using namespace std;
using namespace std::chrono;

template <typename T> __global__ void initKernel(T *data, size_t data_len) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = tidx; idx < data_len; idx += gridDim.x * blockDim.x) {
    data[idx] = idx;
  }
}

// Template version for occupancy calculation (uses N=100 as representative)
template <typename T, int N, int M, int BLOCKSIZE>
__global__ void testfun_template(T *const __restrict__ dA, T *const __restrict__ dB,
                                 T *dC) {
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

void print_usage(const char* prog_name) {
  cerr << "Usage: " << prog_name << " <algorithmic_intensity> <duration_seconds>" << endl;
  cerr << "  algorithmic_intensity: Target arithmetic intensity in Flop/B (e.g., 0.5, 1.0, 2.0)" << endl;
  cerr << "  duration_seconds: How long to run the benchmark in seconds (e.g., 10, 60, 300)" << endl;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    print_usage(argv[0]);
    return 1;
  }

  double target_ai = atof(argv[1]);
  double duration_sec = atof(argv[2]);

  if (target_ai <= 0 || duration_sec <= 0) {
    cerr << "Error: Both algorithmic intensity and duration must be positive numbers" << endl;
    print_usage(argv[0]);
    return 1;
  }

  typedef float dtype;
  const int M = 4000;
  const int BLOCKSIZE = 256;

  // Calculate N from algorithmic intensity
  // AI = (2.0 + N * 2.0) / (2.0 * sizeof(dtype))
  // AI = (2.0 + N * 2.0) / 8.0
  // AI = (1.0 + N) / 4.0
  // 4 * AI = 1.0 + N
  // N = 4 * AI - 1
  int N = (int)round(4.0 * target_ai - 1.0);
  if (N < 0) N = 0;

  double actual_ai = (2.0 + N * 2.0) / (2.0 * sizeof(dtype));

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

    // Use a template kernel with N=100 for occupancy calculation
    // Occupancy is relatively stable across N values for this kernel structure
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, testfun_template<dtype, 100, M, BLOCKSIZE>, BLOCKSIZE, 0));
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

    MeasurementSeries powerSeries;
    MeasurementSeries clockSeries;
    MeasurementSeries temperatureSeries;
    
    size_t iter_count = 0;

    if (omp_get_thread_num() == 0) {
      cout << "Running at algorithmic intensity: " << fixed << setprecision(3) 
           << actual_ai << " Flop/B (N=" << N << ")" << endl;
      cout << "Duration: " << duration_sec << " seconds" << endl;
      cout << "Device " << deviceId << ": Starting..." << endl;
    }

    // Run kernels continuously until time expires
    // Sample stats periodically to avoid affecting kernel timing
    auto last_sample_time = start_time;
    const double sample_interval = 0.1; // Sample every 100ms

    while (steady_clock::now() < end_time) {
      testfun_runtime<dtype, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC, N);
      iter_count++;

      // Sample GPU stats periodically (every ~100ms)
      auto now = steady_clock::now();
      if (duration<double>(now - last_sample_time).count() >= sample_interval) {
        auto stats = getGPUStats(deviceId);
        powerSeries.add(stats.power);
        clockSeries.add(stats.clock);
        temperatureSeries.add(stats.temperature);
        last_sample_time = now;
      }
    }
    
    // Synchronize to ensure all kernels complete
    GPU_ERROR(cudaDeviceSynchronize());

    GPU_ERROR(cudaGetLastError());

    auto actual_duration = duration_cast<milliseconds>(steady_clock::now() - start_time).count() / 1000.0;

#pragma omp barrier
#pragma omp for ordered schedule(static, 1)
    for (int i = 0; i < omp_get_num_threads(); i++) {
#pragma omp ordered
      {
        cout << setprecision(3) << fixed << deviceId << " " << blockCount
             << " blocks   " << setw(3) << N << " its      "
             << actual_ai << " Fl/B      "
             << setprecision(0) << setw(5)
             << iter_count * 2 * data_len * sizeof(dtype) / actual_duration * 1.0e-9
             << " GB/s    " << setw(6)
             << iter_count * (2 + N * 2) * data_len / actual_duration * 1.0e-9 << " GF/s   "
             << clockSeries.median() << " Mhz   "
             << powerSeries.median() / 1000 << " W   "
             << temperatureSeries.median() << "°C   "
             << "Runtime: " << setprecision(1) << actual_duration << "s\n";
      }
    }
    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
    GPU_ERROR(cudaFree(dC));
  }
  cout << "\n";
  return 0;
}

