#include "fnv_hash.h"
#include <cuda_runtime.h>

#define DIVUP(m, n) (((m) + (n)-1) / (n))

/**
 * CUDA kernel: compute FNV-1 hash for each point's voxel coordinate.
 *
 * For each point: floor(coord / voxel_size) → int32 → FNV-1 hash
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
    fnv_hash_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        coord, voxel_size, output_hash, num_points);
}
