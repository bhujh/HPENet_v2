#include "fnv_hash.h"
#include <device_launch_parameters.h>

#define DIVUP(m, n) (((m) + (n)-1) / (n))

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
__host__ __device__ inline uint64_t fnv_hash_coord(int32_t ix, int32_t iy,
    int32_t iz) {
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;

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
 * CUDA kernel: compute FNV-1 hash for each point's voxel coordinate.
 *
 * For each point: floor(coord / voxel_size) ¡ú int32 ¡ú FNV-1 hash
 * (word-level multiply-then-XOR on 64-bit sign-extended values).
 */
__global__ void fnv_hash_kernel(
    const float* __restrict__ coord,
    float voxel_size,
    uint64_t* __restrict__ output_hash,
    int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    const float* pt = coord + idx * 3;
    int32_t ix = static_cast<int32_t>(floorf(pt[0] / voxel_size));
    int32_t iy = static_cast<int32_t>(floorf(pt[1] / voxel_size));
    int32_t iz = static_cast<int32_t>(floorf(pt[2] / voxel_size));

    output_hash[idx] = fnv_hash_coord(ix, iy, iz);
}

void launch_fnv_hash_kernel(
    const float* coord,
    float voxel_size,
    uint64_t* output_hash,
    int num_points,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int grid_size = DIVUP(num_points, BLOCK_SIZE);
    fnv_hash_kernel << <grid_size, BLOCK_SIZE, 0, stream >> > (
        coord, voxel_size, output_hash, num_points);
}
