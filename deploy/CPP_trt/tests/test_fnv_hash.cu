#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include "fnv_hash.h"

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined (set in CMakeLists.txt)"
#endif

// ---------------------------------------------------------------------------
// Helper: CUDA error-checking macro (usable inside TEST/ASSERT scope)
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
        f.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    }
    return data;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class FNVHashTest : public ::testing::Test {
protected:
    static constexpr int    kNumPoints  = 1000;
    static constexpr float  kVoxelSize  = 0.1f;

    float*     d_coord_    = nullptr;
    uint64_t*  d_hash_     = nullptr;
    uint64_t*  h_expected_ = nullptr;

    void SetUp() override {
        const std::string data_dir(TEST_DATA_DIR);

        // --- load golden data from disk ---
        auto h_coord = LoadBin<float>(data_dir + "/fnv_coord.bin", kNumPoints * 3);
        auto h_expected_vec =
            LoadBin<uint64_t>(data_dir + "/fnv_hash.bin", kNumPoints);

        ASSERT_EQ(h_coord.size(), kNumPoints * 3)
            << "Failed to load fnv_coord.bin from " << data_dir;
        ASSERT_EQ(h_expected_vec.size(), kNumPoints)
            << "Failed to load fnv_hash.bin from " << data_dir;

        h_expected_ = new uint64_t[kNumPoints];
        std::copy(h_expected_vec.begin(), h_expected_vec.end(), h_expected_);

        // --- allocate & upload to GPU ---
        CUDA_ASSERT(cudaMalloc(&d_coord_, kNumPoints * 3 * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_hash_, kNumPoints * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMemcpy(d_coord_, h_coord.data(),
                               kNumPoints * 3 * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        cudaFree(d_coord_);
        cudaFree(d_hash_);
        delete[] h_expected_;
    }
};

// ---------------------------------------------------------------------------
// Test: GPU hash matches Python golden reference
// ---------------------------------------------------------------------------
TEST_F(FNVHashTest, MatchesPythonGolden) {
    launch_fnv_hash_kernel(d_coord_, kVoxelSize, d_hash_, kNumPoints);

    CUDA_ASSERT(cudaDeviceSynchronize());

    std::vector<uint64_t> h_result(kNumPoints);
    CUDA_ASSERT(cudaMemcpy(h_result.data(), d_hash_,
                           kNumPoints * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumPoints; ++i) {
        EXPECT_EQ(h_result[i], h_expected_[i])
            << "Hash mismatch at point " << i;
    }
}
