#pragma once
#include <cstdint>

/**
 * @brief FNV-1 64-bit hash for voxel coordinates.
 *
 * Implements word-level FNV-1 (multiply-then-XOR) matching
 * Python's openpoints/dataset/data_util.py::fnv_hash_vec.
 *
 * Each int32 coordinate is sign-extended to 64-bit then
 * reinterpreted as uint64, replicating NumPy's
 * float64.astype(np.uint64) truncation+wrapping semantics.
 *
 * Algorithm per coordinate dimension:
 *   hash = hash * FNV_PRIME ^ uint64_value
 */
__host__ __device__ inline uint64_t fnv_hash_coord(int32_t ix, int32_t iy, int32_t iz) {
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    const uint64_t FNV_PRIME  = 1099511628211ULL;

    // sign-extend to 64-bit, then treat as uint64
    // matches: np.float64(value).astype(np.uint64)
    uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(ix));
    uint64_t uy = static_cast<uint64_t>(static_cast<int64_t>(iy));
    uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(iz));

    uint64_t hash = FNV_OFFSET;
    hash = hash * FNV_PRIME ^ ux;
    hash = hash * FNV_PRIME ^ uy;
    hash = hash * FNV_PRIME ^ uz;
    return hash;
}

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
