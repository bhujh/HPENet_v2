#include "scatter_mean.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: atomicAdd 累加 → sum / count
// ─────────────────────────────────────────────────────────────────────────────
// 每个线程处理一个 (i, c) 元素:
//   - atomicAdd 到 output[g * C + c] 累加该组在该通道的 sum
//   - 当 c == 0 时 atomicAdd 到 count[g]
//
// 负 index 将被跳过。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void scatter_add_kernel(
    const float* __restrict__ src,
    const int64_t* __restrict__ index,
    float* __restrict__ output,
    int* __restrict__ count,
    int num_src, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_src * C) return;
    int i = idx / C;            // source index
    int c = idx % C;            // channel
    int64_t g = index[i];       // group index
    if (g < 0) return;          // skip negative indices

    atomicAdd(&output[g * C + c], src[i * C + c]);
    if (c == 0) {
        atomicAdd(&count[g], 1);  // count once per point
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: 逐元素除以 count[g] 得到均值
// ─────────────────────────────────────────────────────────────────────────────
// 每个线程处理一个 (g, c) 元素，当 count[g] > 0 时做除法。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void scatter_div_kernel(
    float* __restrict__ output,
    const int* __restrict__ count,
    int num_classes, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_classes * C) return;
    int g = idx / C;
    int c = idx % C;
    if (count[g] > 0) {
        output[g * C + c] /= static_cast<float>(count[g]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 启动封装
// ─────────────────────────────────────────────────────────────────────────────
// 使用 cudaMemsetAsync 预先将 output 和 count 置零。
//
// scatter_add_kernel :  num_src * C 个线程, 256 threads/block
// scatter_div_kernel  :  num_classes * C 个线程, 256 threads/block
// ─────────────────────────────────────────────────────────────────────────────
void launch_scatter_mean_kernel(
    const float* src,
    const int64_t* index,
    float* output,
    int* count,
    int num_src,
    int num_classes,
    int C,
    cudaStream_t stream)
{
    // 置零 output 和 count
    cudaMemsetAsync(output, 0, static_cast<size_t>(num_classes) * C * sizeof(float), stream);
    cudaMemsetAsync(count, 0, static_cast<size_t>(num_classes) * sizeof(int), stream);

    // Phase 1: atomicAdd 累加
    {
        const int block_size = 256;
        const int total_threads = num_src * C;
        const int grid_size = (total_threads + block_size - 1) / block_size;
        scatter_add_kernel << <grid_size, block_size, 0, stream >> > (
            src, index, output, count, num_src, C);
    }

    // Phase 2: 除法求均值
    {
        const int block_size = 256;
        const int total_threads = num_classes * C;
        const int grid_size = (total_threads + block_size - 1) / block_size;
        scatter_div_kernel << <grid_size, block_size, 0, stream >> > (
            output, count, num_classes, C);
    }
}
