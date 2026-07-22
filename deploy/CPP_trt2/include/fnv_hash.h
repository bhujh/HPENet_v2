#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>


/**
 * @brief Launch CUDA kernel to batch-compute FNV-1 hashes.
 *
 * @param coord       (N, 3) float32 on device
 * @param voxel_size  Voxel size in meters
 * @param output_hash (N,) uint64 on device (pre-allocated)
 * @param num_points  Number of points
 * @param stream      CUDA stream (default: 0)
 */
void launch_fnv_hash_kernel(
    const float* coord,
    float voxel_size,
    uint64_t* output_hash,
    int num_points,
    cudaStream_t stream = 0
);
