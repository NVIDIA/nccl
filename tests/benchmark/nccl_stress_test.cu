// NCCL Stress Test and Benchmark
// Cross-platform: Windows and Linux
// Usage: nccl_stress_test [num_iterations] [min_size] [max_size]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#define PLATFORM_NAME "Windows"
#else
#include <unistd.h>
#define PLATFORM_NAME "Linux"
#endif

#define CHECK_CUDA(cmd)                                                                  \
    do                                                                                   \
    {                                                                                    \
        cudaError_t e = cmd;                                                             \
        if (e != cudaSuccess)                                                            \
        {                                                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

#define CHECK_NCCL(cmd)                                                                  \
    do                                                                                   \
    {                                                                                    \
        ncclResult_t r = cmd;                                                            \
        if (r != ncclSuccess)                                                            \
        {                                                                                \
            printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

struct BenchmarkResult
{
    const char *operation;
    size_t bytes;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    double bandwidth_gbps;
};

class Timer
{
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

double calculate_std_dev(const std::vector<double> &times, double avg)
{
    double sum_sq = 0.0;
    for (double t : times)
    {
        sum_sq += (t - avg) * (t - avg);
    }
    return std::sqrt(sum_sq / times.size());
}

void print_header()
{
    printf("\n");
    printf("================================================================================\n");
    printf("                    NCCL Stress Test and Benchmark\n");
    printf("                    Platform: %s\n", PLATFORM_NAME);
    printf("================================================================================\n\n");
}

void print_system_info(int nDev)
{
    printf("System Information:\n");
    printf("-------------------\n");

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version:  %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    int ncclMajor, ncclMinor, ncclPatch;
    ncclGetVersion(&ncclMajor);
    // ncclMajor contains full version as XXYYZZ
    printf("  NCCL Version:         %d.%d.%d\n", ncclMajor / 10000, (ncclMajor % 10000) / 100, ncclMajor % 100);

    printf("  Number of GPUs:       %d\n", nDev);

    for (int i = 0; i < nDev; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  GPU %d: %s (SM %d.%d, %.1f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    printf("\n");
}

void print_result(const BenchmarkResult &r)
{
    printf("  %-15s %10zu bytes  %8.3f ms (min: %.3f, max: %.3f, std: %.3f)  %8.2f GB/s\n",
           r.operation, r.bytes, r.avg_time_ms, r.min_time_ms, r.max_time_ms,
           r.std_dev_ms, r.bandwidth_gbps);
}

BenchmarkResult benchmark_allreduce(ncclComm_t *comms, cudaStream_t *streams,
                                    float **buffers, int nDev, size_t count,
                                    int warmup, int iterations)
{
    size_t bytes = count * sizeof(float);
    std::vector<double> times;
    Timer timer;

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclAllReduce(buffers[d], buffers[d], count, ncclFloat, ncclSum, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
    }

    // Benchmark
    for (int i = 0; i < iterations; i++)
    {
        timer.start();
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclAllReduce(buffers[d], buffers[d], count, ncclFloat, ncclSum, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
        times.push_back(timer.stop());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double std_dev = calculate_std_dev(times, avg);
    double bandwidth = (bytes * 2.0 * (nDev - 1) / nDev) / (avg / 1000.0) / 1e9; // Ring algorithm

    return {"AllReduce", bytes, avg, min_t, max_t, std_dev, bandwidth};
}

BenchmarkResult benchmark_broadcast(ncclComm_t *comms, cudaStream_t *streams,
                                    float **buffers, int nDev, size_t count,
                                    int warmup, int iterations)
{
    size_t bytes = count * sizeof(float);
    std::vector<double> times;
    Timer timer;

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclBroadcast(buffers[0], buffers[d], count, ncclFloat, 0, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
    }

    // Benchmark
    for (int i = 0; i < iterations; i++)
    {
        timer.start();
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclBroadcast(buffers[0], buffers[d], count, ncclFloat, 0, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
        times.push_back(timer.stop());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double std_dev = calculate_std_dev(times, avg);
    double bandwidth = bytes / (avg / 1000.0) / 1e9;

    return {"Broadcast", bytes, avg, min_t, max_t, std_dev, bandwidth};
}

BenchmarkResult benchmark_reduce(ncclComm_t *comms, cudaStream_t *streams,
                                 float **buffers, int nDev, size_t count,
                                 int warmup, int iterations)
{
    size_t bytes = count * sizeof(float);
    std::vector<double> times;
    Timer timer;

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclReduce(buffers[d], buffers[0], count, ncclFloat, ncclSum, 0, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
    }

    // Benchmark
    for (int i = 0; i < iterations; i++)
    {
        timer.start();
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclReduce(buffers[d], buffers[0], count, ncclFloat, ncclSum, 0, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
        times.push_back(timer.stop());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double std_dev = calculate_std_dev(times, avg);
    double bandwidth = bytes / (avg / 1000.0) / 1e9;

    return {"Reduce", bytes, avg, min_t, max_t, std_dev, bandwidth};
}

BenchmarkResult benchmark_allgather(ncclComm_t *comms, cudaStream_t *streams,
                                    float **sendbufs, float **recvbufs, int nDev, size_t count,
                                    int warmup, int iterations)
{
    size_t bytes = count * sizeof(float);
    std::vector<double> times;
    Timer timer;

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclAllGather(sendbufs[d], recvbufs[d], count, ncclFloat, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
    }

    // Benchmark
    for (int i = 0; i < iterations; i++)
    {
        timer.start();
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclAllGather(sendbufs[d], recvbufs[d], count, ncclFloat, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
        times.push_back(timer.stop());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double std_dev = calculate_std_dev(times, avg);
    double bandwidth = (bytes * (nDev - 1)) / (avg / 1000.0) / 1e9;

    return {"AllGather", bytes, avg, min_t, max_t, std_dev, bandwidth};
}

BenchmarkResult benchmark_reducescatter(ncclComm_t *comms, cudaStream_t *streams,
                                        float **sendbufs, float **recvbufs, int nDev, size_t count,
                                        int warmup, int iterations)
{
    size_t bytes = count * sizeof(float);
    std::vector<double> times;
    Timer timer;

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclReduceScatter(sendbufs[d], recvbufs[d], count, ncclFloat, ncclSum, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
    }

    // Benchmark
    for (int i = 0; i < iterations; i++)
    {
        timer.start();
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclReduceScatter(sendbufs[d], recvbufs[d], count, ncclFloat, ncclSum, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
        times.push_back(timer.stop());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double std_dev = calculate_std_dev(times, avg);
    double bandwidth = (bytes * (nDev - 1)) / (avg / 1000.0) / 1e9;

    return {"ReduceScatter", bytes, avg, min_t, max_t, std_dev, bandwidth};
}

BenchmarkResult benchmark_sendrecv(ncclComm_t *comms, cudaStream_t *streams,
                                   float **sendbufs, float **recvbufs, int nDev, size_t count,
                                   int warmup, int iterations)
{
    size_t bytes = count * sizeof(float);
    std::vector<double> times;
    Timer timer;

    // Ring pattern: GPU i sends to GPU (i+1)%nDev and receives from GPU (i-1+nDev)%nDev

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            int sendPeer = (d + 1) % nDev;
            int recvPeer = (d - 1 + nDev) % nDev;
            CHECK_NCCL(ncclSend(sendbufs[d], count, ncclFloat, sendPeer, comms[d], streams[d]));
            CHECK_NCCL(ncclRecv(recvbufs[d], count, ncclFloat, recvPeer, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
    }

    // Benchmark
    for (int i = 0; i < iterations; i++)
    {
        timer.start();
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            int sendPeer = (d + 1) % nDev;
            int recvPeer = (d - 1 + nDev) % nDev;
            CHECK_NCCL(ncclSend(sendbufs[d], count, ncclFloat, sendPeer, comms[d], streams[d]));
            CHECK_NCCL(ncclRecv(recvbufs[d], count, ncclFloat, recvPeer, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        }
        times.push_back(timer.stop());
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double std_dev = calculate_std_dev(times, avg);
    double bandwidth = (bytes * nDev * 2) / (avg / 1000.0) / 1e9; // bidirectional

    return {"SendRecv", bytes, avg, min_t, max_t, std_dev, bandwidth};
}

int main(int argc, char *argv[])
{
    int iterations = 100;
    size_t min_size = 1024;              // 1 KB
    size_t max_size = 128 * 1024 * 1024; // 128 MB

    if (argc > 1)
        iterations = atoi(argv[1]);
    if (argc > 2)
        min_size = atol(argv[2]);
    if (argc > 3)
        max_size = atol(argv[3]);

    int warmup = 10;

    print_header();

    // Get device count
    int nDev;
    CHECK_CUDA(cudaGetDeviceCount(&nDev));
    if (nDev < 2)
    {
        printf("Error: At least 2 GPUs required for this benchmark\n");
        return 1;
    }

    print_system_info(nDev);

    printf("Benchmark Configuration:\n");
    printf("------------------------\n");
    printf("  Iterations: %d\n", iterations);
    printf("  Warmup:     %d\n", warmup);
    printf("  Min size:   %zu bytes\n", min_size);
    printf("  Max size:   %zu bytes\n", max_size);
    printf("\n");

    // Initialize NCCL
    std::vector<int> devs(nDev);
    for (int i = 0; i < nDev; i++)
        devs[i] = i;

    std::vector<ncclComm_t> comms(nDev);
    std::vector<cudaStream_t> streams(nDev);

    printf("Initializing NCCL communicators...\n");
    CHECK_NCCL(ncclCommInitAll(comms.data(), nDev, devs.data()));
    printf("NCCL initialized successfully!\n\n");

    // Create streams
    for (int d = 0; d < nDev; d++)
    {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaStreamCreate(&streams[d]));
    }

    // Test different sizes
    std::vector<size_t> sizes;
    for (size_t s = min_size; s <= max_size; s *= 4)
    {
        sizes.push_back(s);
    }

    printf("================================================================================\n");
    printf("                           BENCHMARK RESULTS\n");
    printf("================================================================================\n");

    for (size_t bytes : sizes)
    {
        size_t count = bytes / sizeof(float);

        printf("\n--- Message Size: %zu bytes (%.2f MB) ---\n", bytes, bytes / (1024.0 * 1024.0));

        // Allocate buffers
        std::vector<float *> sendbufs(nDev), recvbufs(nDev), largebufs(nDev);
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaMalloc(&sendbufs[d], bytes));
            CHECK_CUDA(cudaMalloc(&recvbufs[d], bytes));
            CHECK_CUDA(cudaMalloc(&largebufs[d], bytes * nDev)); // For AllGather
            CHECK_CUDA(cudaMemset(sendbufs[d], 1, bytes));
            CHECK_CUDA(cudaMemset(recvbufs[d], 0, bytes));
        }

        // Run benchmarks
        print_result(benchmark_allreduce(comms.data(), streams.data(), sendbufs.data(), nDev, count, warmup, iterations));
        print_result(benchmark_broadcast(comms.data(), streams.data(), sendbufs.data(), nDev, count, warmup, iterations));
        print_result(benchmark_reduce(comms.data(), streams.data(), sendbufs.data(), nDev, count, warmup, iterations));
        print_result(benchmark_allgather(comms.data(), streams.data(), sendbufs.data(), largebufs.data(), nDev, count, warmup, iterations));
        print_result(benchmark_reducescatter(comms.data(), streams.data(), largebufs.data(), recvbufs.data(), nDev, count, warmup, iterations));
        // SendRecv disabled due to WSL2 P2P issues
        // print_result(benchmark_sendrecv(comms.data(), streams.data(), sendbufs.data(), recvbufs.data(), nDev, count, warmup, iterations));

        // Free buffers
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_CUDA(cudaFree(sendbufs[d]));
            CHECK_CUDA(cudaFree(recvbufs[d]));
            CHECK_CUDA(cudaFree(largebufs[d]));
        }
    }

    printf("\n================================================================================\n");
    printf("                           STRESS TEST\n");
    printf("================================================================================\n");

    // Stress test: rapid fire operations
    size_t stress_size = 1024 * 1024; // 1 MB
    size_t stress_count = stress_size / sizeof(float);
    int stress_iterations = 1000;

    printf("\nRunning stress test: %d rapid iterations with %zu byte messages...\n",
           stress_iterations, stress_size);

    std::vector<float *> stress_bufs(nDev);
    for (int d = 0; d < nDev; d++)
    {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaMalloc(&stress_bufs[d], stress_size));
    }

    Timer stress_timer;
    stress_timer.start();

    for (int i = 0; i < stress_iterations; i++)
    {
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < nDev; d++)
        {
            CHECK_CUDA(cudaSetDevice(d));
            CHECK_NCCL(ncclAllReduce(stress_bufs[d], stress_bufs[d], stress_count,
                                     ncclFloat, ncclSum, comms[d], streams[d]));
        }
        CHECK_NCCL(ncclGroupEnd());
    }

    // Final sync
    for (int d = 0; d < nDev; d++)
    {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaStreamSynchronize(streams[d]));
    }

    double stress_time = stress_timer.stop();
    double ops_per_sec = stress_iterations / (stress_time / 1000.0);
    double total_data = (double)stress_size * stress_iterations * nDev;
    double throughput = total_data / (stress_time / 1000.0) / 1e9;

    printf("  Stress test completed!\n");
    printf("  Total time:       %.2f seconds\n", stress_time / 1000.0);
    printf("  Operations/sec:   %.2f\n", ops_per_sec);
    printf("  Total data:       %.2f GB\n", total_data / 1e9);
    printf("  Throughput:       %.2f GB/s\n", throughput);

    // Cleanup
    for (int d = 0; d < nDev; d++)
    {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaFree(stress_bufs[d]));
        CHECK_CUDA(cudaStreamDestroy(streams[d]));
        CHECK_NCCL(ncclCommDestroy(comms[d]));
    }

    printf("\n================================================================================\n");
    printf("                    BENCHMARK COMPLETE - %s\n", PLATFORM_NAME);
    printf("================================================================================\n");

    return 0;
}
