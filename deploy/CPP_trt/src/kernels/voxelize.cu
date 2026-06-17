#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include "fnv_hash.h"

// ===========================================================================
// launch_voxelize_sort  —  GPU hash + sort pipeline
//
// Step 1: launch FNV-1 64-bit hash kernel for all points
// Step 2: create index sequence [0, 1, ..., N-1]
// Step 3: sort indices by hash (in-place on device arrays)
//
// After this function:
//   d_hashes_sorted[0..N-1]  = hashes sorted ascending
//   d_indices_sorted[0..N-1] = original point indices in hash order
//
// All operations use the default CUDA stream (stream 0) so they are
// implicitly serialised — no explicit synchronisation needed.
// ===========================================================================
void launch_voxelize_sort(
    const float* d_coord,
    float voxel_size,
    int num_points,
    uint64_t* d_hashes_sorted,
    int64_t* d_indices_sorted)
{
    // ── Step 1: compute FNV hash per point ────────────────────────────
    launch_fnv_hash_kernel(d_coord, voxel_size, d_hashes_sorted, num_points, 0);

    // ── Step 2: create index sequence [0, 1, ..., N-1] ────────────────
    thrust::device_ptr<int64_t> d_idx(d_indices_sorted);
    thrust::sequence(d_idx, d_idx + num_points, 0);

    // ── Step 3: sort indices by hash value (keys in-place) ────────────
    thrust::device_ptr<uint64_t> d_hash(d_hashes_sorted);
    thrust::sort_by_key(d_hash, d_hash + num_points, d_idx);
}
