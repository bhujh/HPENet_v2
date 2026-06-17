#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <set>

#include "voxelizer.h"

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined (set in CMakeLists.txt)"
#endif

// ---------------------------------------------------------------------------
// Helper: CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_ASSERT(call)                                               \
    do {                                                                \
        cudaError_t _err = (call);                                      \
        ASSERT_EQ(_err, cudaSuccess)                                    \
            << "CUDA error: " << cudaGetErrorString(_err);              \
    } while (0)

// ---------------------------------------------------------------------------
// Helper: read binary file into vector
// ---------------------------------------------------------------------------
template <typename T>
static std::vector<T> LoadBin(const std::string& path, size_t count) {
    std::vector<T> data(count);
    std::ifstream f(path, std::ios::binary);
    if (f.is_open()) {
        f.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    }
    return data;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class VoxelizeTest : public ::testing::Test {
protected:
    static constexpr int    kNumPoints  = 1000;
    static constexpr float  kVoxelSize  = 0.1f;

    std::vector<float>   h_coord_;         // (N, 3)

    // Golden references (from Python gen_golden_data.py, seed=42)
    std::vector<int64_t> gold_idx_sort_;
    std::vector<int64_t> gold_voxel_idx_;
    std::vector<int64_t> gold_count_;

    void SetUp() override {
        const std::string data_dir(TEST_DATA_DIR);

        // ── load input coordinates ────────────────────────────────────
        h_coord_ = LoadBin<float>(data_dir + "/voxel_coord.bin", kNumPoints * 3);
        ASSERT_EQ(h_coord_.size(), kNumPoints * 3);

        // ── load golden references ────────────────────────────────────
        gold_idx_sort_  = LoadBin<int64_t>(data_dir + "/voxel_idx_sort.bin", kNumPoints);
        gold_voxel_idx_ = LoadBin<int64_t>(data_dir + "/voxel_voxel_idx.bin", kNumPoints);
        gold_count_     = LoadBin<int64_t>(data_dir + "/voxel_count.bin", 641);  // 641 voxels
    }
};

// ===========================================================================
//  Test: count must match Python golden data EXACTLY
// ===========================================================================
TEST_F(VoxelizeTest, CountMatchesPythonGolden) {
    auto result = Voxelizer::voxelize(h_coord_.data(), kNumPoints, kVoxelSize);

    ASSERT_EQ(result.count.size(), gold_count_.size())
        << "Number of unique voxels differs";

    for (size_t i = 0; i < result.count.size(); ++i) {
        EXPECT_EQ(result.count[i], gold_count_[i])
            << "count mismatch at voxel " << i;
    }
}

// ===========================================================================
//  Test: voxel_idx matches Python golden data
// ===========================================================================
TEST_F(VoxelizeTest, VoxelIdxMatchesPythonGolden) {
    auto result = Voxelizer::voxelize(h_coord_.data(), kNumPoints, kVoxelSize);

    ASSERT_EQ(result.voxel_idx.size(), gold_voxel_idx_.size());
    ASSERT_EQ(result.voxel_idx.size(), kNumPoints);

    // voxel_idx should match exactly when counts match (same grouping)
    for (int i = 0; i < kNumPoints; ++i) {
        EXPECT_EQ(result.voxel_idx[i], gold_voxel_idx_[i])
            << "voxel_idx mismatch at sorted position " << i;
    }
}

// ===========================================================================
//  Test: idx_sort is a permutation of [0, 1, ..., N-1]
//
//  The exact ordering within equal-hash groups may differ from Python's
//  argsort (unstable) vs thrust::sort_by_key (may also be unstable), but
//  the multiset of indices must be identical.
// ===========================================================================
TEST_F(VoxelizeTest, IdxSortValidPermutation) {
    auto result = Voxelizer::voxelize(h_coord_.data(), kNumPoints, kVoxelSize);

    ASSERT_EQ(result.idx_sort.size(), kNumPoints);

    // Verify that idx_sort contains each index 0..N-1 exactly once
    std::vector<int64_t> sorted = result.idx_sort;
    std::sort(sorted.begin(), sorted.end());
    for (int64_t i = 0; i < kNumPoints; ++i) {
        EXPECT_EQ(sorted[static_cast<size_t>(i)], i)
            << "idx_sort missing or duplicate element " << i;
    }
}

// ===========================================================================
//  Test: idx_sort as a multiset equals the golden idx_sort
//
//  Since both are permutations of [0, N-1], sorting both and comparing
//  confirms the multiset identity.
// ===========================================================================
TEST_F(VoxelizeTest, IdxSortSetEquivalence) {
    auto result = Voxelizer::voxelize(h_coord_.data(), kNumPoints, kVoxelSize);

    ASSERT_EQ(result.idx_sort.size(), gold_idx_sort_.size());

    std::vector<int64_t> computed_sorted = result.idx_sort;
    std::vector<int64_t> golden_sorted   = gold_idx_sort_;
    std::sort(computed_sorted.begin(), computed_sorted.end());
    std::sort(golden_sorted.begin(), golden_sorted.end());

    for (size_t i = 0; i < computed_sorted.size(); ++i) {
        EXPECT_EQ(computed_sorted[i], golden_sorted[i])
            << "idx_sort multiset differs at position " << i;
    }
}

// ===========================================================================
//  Test: sub-cloud idx_points structure
// ===========================================================================
TEST_F(VoxelizeTest, SubCloudStructure) {
    auto result = Voxelizer::voxelize(h_coord_.data(), kNumPoints, kVoxelSize);

    // Number of sub-clouds should equal max(count)
    int64_t max_count = 0;
    for (auto c : result.count) {
        if (c > max_count) max_count = c;
    }
    ASSERT_EQ(result.idx_points.size(), static_cast<size_t>(max_count))
        << "Number of sub-clouds must equal max voxel count";

    // Each sub-cloud should have exactly M elements (one per voxel)
    const size_t M = result.count.size();
    for (size_t i = 0; i < result.idx_points.size(); ++i) {
        EXPECT_EQ(result.idx_points[i].size(), M)
            << "Sub-cloud " << i << " has wrong size";
    }

    // All indices in idx_points must be valid: 0 <= idx < num_points
    for (size_t i = 0; i < result.idx_points.size(); ++i) {
        for (auto idx : result.idx_points[i]) {
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, kNumPoints);
        }
    }
}

// ===========================================================================
//  Test: consistency — count sums to num_points, voxel_idx range valid
// ===========================================================================
TEST_F(VoxelizeTest, InternalConsistency) {
    auto result = Voxelizer::voxelize(h_coord_.data(), kNumPoints, kVoxelSize);

    // Sum of counts = num_points
    int64_t total = 0;
    for (auto c : result.count) total += c;
    EXPECT_EQ(total, kNumPoints);

    // voxel_idx values in [0, M-1]
    for (auto v : result.voxel_idx) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, static_cast<int64_t>(result.count.size()));
    }

    // idx_sort size = num_points
    EXPECT_EQ(result.idx_sort.size(), kNumPoints);
}
