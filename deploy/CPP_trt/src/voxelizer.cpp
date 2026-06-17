#include "voxelizer.h"
#include "random_util.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

// ===========================================================================
//  GPU function from src/kernels/voxelize.cu — declared in voxelizer.h
// ===========================================================================

// ===========================================================================
//  Voxelizer::voxelize
//
//  GPU + CPU hybrid pipeline:
//    1. Upload coordinates → GPU
//    2. Compute FNV hashes + thrust::sort_by_key   (GPU)
//    3. Download sorted hashes + indices            (GPU → CPU)
//    4. Scan sorted hashes → unique + counts        (CPU)
//    5. Build sub-cloud idx_points with shuffle     (CPU)
//
//  Matches Python:
//    openpoints/dataset/data_util.py:voxelize(mode=1)
//    deploy/common.py:preprocess_test
// ===========================================================================
VoxelizeResult Voxelizer::voxelize(
    const float* coord,
    int num_points,
    float voxel_size,
    int seed)
{
    if (num_points <= 0) {
        return VoxelizeResult();
    }

    const size_t coord_bytes  = static_cast<size_t>(num_points) * 3 * sizeof(float);
    const size_t hash_bytes   = static_cast<size_t>(num_points) * sizeof(uint64_t);
    const size_t index_bytes  = static_cast<size_t>(num_points) * sizeof(int64_t);

    // ── allocate GPU memory ───────────────────────────────────────────
    float*   d_coord   = nullptr;
    uint64_t* d_hashes  = nullptr;
    int64_t* d_indices  = nullptr;

    auto free_all = [&]() {
        cudaFree(d_coord);
        cudaFree(d_hashes);
        cudaFree(d_indices);
    };

    cudaError_t err;

    err = cudaMalloc(&d_coord,  coord_bytes);
    if (err) { free_all(); throw std::runtime_error("Voxelizer cudaMalloc coord: "
                               + std::string(cudaGetErrorString(err))); }
    err = cudaMalloc(&d_hashes, hash_bytes);
    if (err) { free_all(); throw std::runtime_error("Voxelizer cudaMalloc hashes: "
                               + std::string(cudaGetErrorString(err))); }
    err = cudaMalloc(&d_indices, index_bytes);
    if (err) { free_all(); throw std::runtime_error("Voxelizer cudaMalloc indices: "
                               + std::string(cudaGetErrorString(err))); }

    // ── upload coordinates ────────────────────────────────────────────
    err = cudaMemcpy(d_coord, coord, coord_bytes, cudaMemcpyHostToDevice);
    if (err) { free_all(); throw std::runtime_error("Voxelizer cudaMemcpy coord: "
                               + std::string(cudaGetErrorString(err))); }

    // ── GPU: hash + sort ──────────────────────────────────────────────
    launch_voxelize_sort(d_coord, voxel_size, num_points, d_hashes, d_indices);

    // ── download results ──────────────────────────────────────────────
    std::vector<uint64_t> hashes_sorted(num_points);
    std::vector<int64_t>  indices_sorted(num_points);

    err = cudaMemcpy(hashes_sorted.data(),  d_hashes,  hash_bytes,  cudaMemcpyDeviceToHost);
    if (err) { free_all(); throw std::runtime_error("Voxelizer cudaMemcpy hashes: "
                               + std::string(cudaGetErrorString(err))); }
    err = cudaMemcpy(indices_sorted.data(), d_indices, index_bytes, cudaMemcpyDeviceToHost);
    if (err) { free_all(); throw std::runtime_error("Voxelizer cudaMemcpy indices: "
                               + std::string(cudaGetErrorString(err))); }

    // ── free GPU memory (safe to call on nullptr) ─────────────────────
    free_all();

    // ══════════════════════════════════════════════════════════════════
    //  CPU: unique + counts
    // ══════════════════════════════════════════════════════════════════

    VoxelizeResult result;
    result.idx_sort = std::move(indices_sorted);

    // Walk through sorted hashes — each hash change = new voxel
    std::vector<int64_t> count;
    std::vector<int64_t> voxel_idx(num_points);

    int64_t cur_voxel       = 0;
    int64_t cur_count       = 1;
    voxel_idx[0]            = 0;

    for (int i = 1; i < num_points; ++i) {
        if (hashes_sorted[i] == hashes_sorted[i - 1]) {
            ++cur_count;
        } else {
            count.push_back(cur_count);
            cur_count = 1;
            ++cur_voxel;
        }
        voxel_idx[i] = cur_voxel;
    }
    count.push_back(cur_count);   // last voxel

    result.voxel_idx = std::move(voxel_idx);
    result.count     = std::move(count);

    // ══════════════════════════════════════════════════════════════════
    //  CPU: sub-cloud split (matches deploy/common.py:preprocess_test)
    // ══════════════════════════════════════════════════════════════════
    //
    //  Python reference:
    //    for i in range(count.max()):
    //        idx_select = cumsum(insert(count,0,0)[0:-1]) + i % count
    //        idx_part = idx_sort[idx_select]
    //        np.random.shuffle(idx_part)
    //        idx_points.append(idx_part)
    // ══════════════════════════════════════════════════════════════════

    const auto& cnt        = result.count;
    const auto& idx_s      = result.idx_sort;
    const int64_t M        = static_cast<int64_t>(cnt.size());

    int64_t max_count = 0;
    for (auto c : cnt) {
        if (c > max_count) max_count = c;
    }

    // Build prefix sums: prefix[0] = 0, prefix[j] = sum(cnt[0..j-1])
    std::vector<int64_t> prefix(M + 1, 0);
    for (int64_t j = 0; j < M; ++j) {
        prefix[j + 1] = prefix[j] + cnt[j];
    }

    // Single RNG instance — state advances through all shuffles
    NumpyMT19937 rng(static_cast<uint32_t>(seed));

    for (int64_t i = 0; i < max_count; ++i) {
        std::vector<int> idx_part(static_cast<size_t>(M));
        for (int64_t j = 0; j < M; ++j) {
            // idx_select[j] = prefix[j] + i % cnt[j]
            int64_t select_idx = prefix[j] + (i % cnt[j]);
            idx_part[static_cast<size_t>(j)] = static_cast<int>(idx_s[select_idx]);
        }
        rng.shuffle(idx_part.data(), static_cast<int>(M));
        result.idx_points.push_back(std::move(idx_part));
    }

    return result;
}
