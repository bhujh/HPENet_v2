#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "scatter_mean.h"

// ─────────────────────────────────────────────────────────────────────────────
// 辅助: 加载二进制文件到 vector
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static std::vector<T> load_bin(const std::string& path, size_t expected_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open: " + path);
    }
    f.seekg(0, std::ios::end);
    size_t file_bytes = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    EXPECT_EQ(file_bytes, expected_bytes)
        << "File " << path << " size mismatch: expected " << expected_bytes
        << " bytes, got " << file_bytes << " bytes";
    std::vector<T> data(file_bytes / sizeof(T));
    f.read(reinterpret_cast<char*>(data.data()), file_bytes);
    return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: 获取测试数据目录 (相对于可执行文件路径)
// ─────────────────────────────────────────────────────────────────────────────
static std::string data_dir() {
    // ctest 将工作目录设为 build/tests/ (add_test 默认 WORKING_DIRECTORY)
    // 所以数据路径为 data/
    return "data/";
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试 1: 黄金数据对比
// ─────────────────────────────────────────────────────────────────────────────
// 加载 scatter_src.bin (64×2 float32), scatter_idx.bin (64 int64),
// scatter_result.bin (5×2 float32), 在 GPU 上执行 scatter_mean,
// 下载结果并与黄金数据对比, 逐元素误差 < 1e-4。
// ─────────────────────────────────────────────────────────────────────────────
TEST(ScatterMeanTest, GoldenData) {
    const int N = 64;
    const int C = 2;
    const int num_classes = 5;

    // 加载黄金数据
    auto src_host   = load_bin<float>(  data_dir() + "scatter_src.bin",    static_cast<size_t>(N) * C * sizeof(float));
    auto idx_host   = load_bin<int64_t>(data_dir() + "scatter_idx.bin",   static_cast<size_t>(N)     * sizeof(int64_t));
    auto ref_host   = load_bin<float>(  data_dir() + "scatter_result.bin", static_cast<size_t>(num_classes) * C * sizeof(float));

    // 分配 GPU 内存
    float*   src_dev   = nullptr;
    int64_t* idx_dev   = nullptr;
    float*   out_dev   = nullptr;
    int*     cnt_dev   = nullptr;

    cudaMalloc(&src_dev, static_cast<size_t>(N) * C * sizeof(float));
    cudaMalloc(&idx_dev, static_cast<size_t>(N)     * sizeof(int64_t));
    cudaMalloc(&out_dev, static_cast<size_t>(num_classes) * C * sizeof(float));
    cudaMalloc(&cnt_dev, static_cast<size_t>(num_classes)     * sizeof(int));

    // 拷贝输入到 GPU
    cudaMemcpy(src_dev, src_host.data(), static_cast<size_t>(N) * C * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(idx_dev, idx_host.data(), static_cast<size_t>(N)     * sizeof(int64_t), cudaMemcpyHostToDevice);

    // 执行 scatter_mean
    launch_scatter_mean_kernel(src_dev, idx_dev, out_dev, cnt_dev, N, num_classes, C, 0);

    // 下载结果
    std::vector<float> out_host(static_cast<size_t>(num_classes) * C);
    cudaMemcpy(out_host.data(), out_dev, static_cast<size_t>(num_classes) * C * sizeof(float), cudaMemcpyDeviceToHost);

    // 对比黄金数据
    float max_err = 0.0f;
    for (int i = 0; i < num_classes * C; ++i) {
        float err = std::fabs(out_host[i] - ref_host[i]);
        max_err = std::max(max_err, err);
        EXPECT_NEAR(out_host[i], ref_host[i], 1e-4)
            << "Mismatch at index " << i << " (class=" << i / C << ", channel=" << i % C << ")"
            << ": got " << out_host[i] << ", expected " << ref_host[i];
    }

    // 输出最大误差方便调试
    std::cout << "[GoldenData] max absolute error = " << max_err << " (threshold: 1e-4)" << std::endl;

    // 清理
    cudaFree(src_dev);
    cudaFree(idx_dev);
    cudaFree(out_dev);
    cudaFree(cnt_dev);
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试 2: 空桶测试
// ─────────────────────────────────────────────────────────────────────────────
// 创建 index 包含跳过的组号 (e.g., 索引 0, 2, 4 有数据, 1, 3 无数据),
// 验证空桶输出为 0。
// ─────────────────────────────────────────────────────────────────────────────
TEST(ScatterMeanTest, EmptyBucket) {
    const int N = 4;
    const int C = 2;
    const int num_classes = 5;  // classes 0..4, 其中 1 和 3 为空

    // src: 4 个点, 每个点 2 维
    std::vector<float> src_host = {
        1.0f, 2.0f,   // idx=0 → class 0
        3.0f, 4.0f,   // idx=0 → class 0
        5.0f, 6.0f,   // idx=2 → class 2
        7.0f, 8.0f    // idx=4 → class 4
    };
    // index: classes 0, 0, 2, 4 (1 和 3 为空)
    std::vector<int64_t> idx_host = {0, 0, 2, 4};

    // 预期结果: class 0 均值 = (1+3)/2=2, (2+4)/2=3
    //          class 1 = 0, 0
    //          class 2 = 5, 6
    //          class 3 = 0, 0
    //          class 4 = 7, 8
    std::vector<float> expected = {
        2.0f, 3.0f,   // class 0
        0.0f, 0.0f,   // class 1 (empty)
        5.0f, 6.0f,   // class 2
        0.0f, 0.0f,   // class 3 (empty)
        7.0f, 8.0f    // class 4
    };

    // GPU 内存
    float*   src_dev = nullptr;
    int64_t* idx_dev = nullptr;
    float*   out_dev = nullptr;
    int*     cnt_dev = nullptr;

    cudaMalloc(&src_dev, static_cast<size_t>(N) * C * sizeof(float));
    cudaMalloc(&idx_dev, static_cast<size_t>(N)     * sizeof(int64_t));
    cudaMalloc(&out_dev, static_cast<size_t>(num_classes) * C * sizeof(float));
    cudaMalloc(&cnt_dev, static_cast<size_t>(num_classes)     * sizeof(int));

    cudaMemcpy(src_dev, src_host.data(), static_cast<size_t>(N) * C * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(idx_dev, idx_host.data(), static_cast<size_t>(N)     * sizeof(int64_t), cudaMemcpyHostToDevice);

    launch_scatter_mean_kernel(src_dev, idx_dev, out_dev, cnt_dev, N, num_classes, C, 0);

    std::vector<float> out_host(static_cast<size_t>(num_classes) * C);
    cudaMemcpy(out_host.data(), out_dev, static_cast<size_t>(num_classes) * C * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < num_classes * C; ++i) {
        float err = std::fabs(out_host[i] - expected[i]);
        max_err = std::max(max_err, err);
        EXPECT_NEAR(out_host[i], expected[i], 1e-5)
            << "EmptyBucket: Mismatch at index " << i;
    }
    std::cout << "[EmptyBucket] max absolute error = " << max_err << std::endl;

    cudaFree(src_dev);
    cudaFree(idx_dev);
    cudaFree(out_dev);
    cudaFree(cnt_dev);
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试 3: 负索引跳过
// ─────────────────────────────────────────────────────────────────────────────
// 确保 index 中的负值被跳过 (不影响结果),
// 对比排除负索引后的参考结果。
// ─────────────────────────────────────────────────────────────────────────────
TEST(ScatterMeanTest, NegativeIndex) {
    const int N = 5;
    const int C = 2;
    const int num_classes = 3;

    // src: 5 个点
    std::vector<float> src_host = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f
    };
    // index: -1 (skip), 0, -1 (skip), 1, 2
    std::vector<int64_t> idx_host = {-1, 0, -1, 1, 2};

    // 预期: class 0 = [3, 4]; class 1 = [7, 8]; class 2 = [9, 10]
    std::vector<float> expected = {
        3.0f, 4.0f,
        7.0f, 8.0f,
        9.0f, 10.0f
    };

    float*   src_dev = nullptr;
    int64_t* idx_dev = nullptr;
    float*   out_dev = nullptr;
    int*     cnt_dev = nullptr;

    cudaMalloc(&src_dev, static_cast<size_t>(N) * C * sizeof(float));
    cudaMalloc(&idx_dev, static_cast<size_t>(N)     * sizeof(int64_t));
    cudaMalloc(&out_dev, static_cast<size_t>(num_classes) * C * sizeof(float));
    cudaMalloc(&cnt_dev, static_cast<size_t>(num_classes)     * sizeof(int));

    cudaMemcpy(src_dev, src_host.data(), static_cast<size_t>(N) * C * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(idx_dev, idx_host.data(), static_cast<size_t>(N)     * sizeof(int64_t), cudaMemcpyHostToDevice);

    launch_scatter_mean_kernel(src_dev, idx_dev, out_dev, cnt_dev, N, num_classes, C, 0);

    std::vector<float> out_host(static_cast<size_t>(num_classes) * C);
    cudaMemcpy(out_host.data(), out_dev, static_cast<size_t>(num_classes) * C * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < num_classes * C; ++i) {
        float err = std::fabs(out_host[i] - expected[i]);
        max_err = std::max(max_err, err);
        EXPECT_NEAR(out_host[i], expected[i], 1e-5)
            << "NegativeIndex: Mismatch at index " << i;
    }
    std::cout << "[NegativeIndex] max absolute error = " << max_err << std::endl;

    cudaFree(src_dev);
    cudaFree(idx_dev);
    cudaFree(out_dev);
    cudaFree(cnt_dev);
}
