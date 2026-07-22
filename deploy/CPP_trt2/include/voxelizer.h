#pragma once
//#include <vector>
//#include <cstdint>
#include "types.h"
#include "random_util.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
/**
 * @brief Result of GPU-accelerated voxelization.
 *
 * Produced by Voxelizer::voxelize, matching the output of
 * Python's openpoints/dataset/data_util.py:voxelize(mode=1) +
 * deploy/common.py:preprocess_test sub-cloud splitting.
 */
struct VoxelizeResult {
    /// Sorted point indices (N,) — indices into original coord array,
    /// ordered by ascending FNV hash value.
    std::vector<int64_t> idx_sort;

    /// Voxel label per point (N,) — each point is assigned a label
    /// 0..M-1 indicating which voxel it belongs to, where M = number
    /// of unique voxels.  Points with the same hash share the same label.
    std::vector<int64_t> voxel_idx;

    /// Points per voxel (M,) — count[j] is the number of points in voxel j.
    std::vector<int64_t> count;

    /// Sub-cloud index lists — count.max() lists, each containing one
    /// representative point per voxel (cycling with modulo), shuffled
    /// with NumpyMT19937 seeded by @p seed.
    /// Matches deploy/common.py:preprocess_test.
    std::vector<std::vector<int>> idx_points;
};

/**
 * @brief GPU-accelerated voxelizer: FNV hash → sort → unique → sub-cloud split.
 *
 * Algorithm (matching Python reference):
 *   1. Discrete coordinates: floor(coord / voxel_size)
 *   2. FNV-1 64-bit hash per point (GPU kernel)
 *   3. thrust::sort_by_key: sort indices by hash (GPU)
 *   4. Host: scan sorted hashes → unique + counts
 *   5. Host: split into sub-clouds with NumpyMT19937 shuffle
 */
class Voxelizer {
public:
    /**
     * @brief Voxelize a point cloud.
     *
     * @param coord       (N, 3) float32 row-major coordinates.
     * @param num_points  Number of points N.
     * @param voxel_size  Voxel size in meters (e.g. 0.1).
     * @param seed        Random seed for sub-cloud shuffle (default 100).
     * @return VoxelizeResult  containing idx_sort, voxel_idx, count, idx_points.
     */
    static VoxelizeResult voxelize(
        const float* coord,
        int num_points,
        float voxel_size,
        int seed = 100);
};

// ---------------------------------------------------------------------------
// Internal: GPU function declared here so both voxelizer.cpp (C++)
// and voxelize.cu (CUDA) can reference it.
// Implemented in src/kernels/voxelize.cu.
// ---------------------------------------------------------------------------

/**
 * @brief Compute FNV hashes and sort indices by hash on GPU.
 *
 * @param d_coord          (N, 3) float32 on device
 * @param voxel_size       Voxel size in meters
 * @param num_points       Number of points
 * @param d_hashes_sorted  (N,) uint64 on device — output: sorted hashes
 * @param d_indices_sorted (N,) int64 on device — output: sorted original indices
 */
void launch_voxelize_sort(
    const float* d_coord,
    float voxel_size,
    int num_points,
    uint64_t* d_hashes_sorted,
    int64_t* d_indices_sorted);
