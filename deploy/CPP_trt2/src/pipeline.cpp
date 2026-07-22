// ============================================================================
// HPENet V2 — InferencePipeline: 完整推理流水线实现
// 流程 (匹配 Python deploy/trt_inference.py:infer_one_cloud_trt + main):
//   1. PlyReader::load → PointCloud
//   2. coord -= min(coord)  (移至原点)
//   3. Voxelizer::voxelize → idx_points (子云列表)
//   4. for each subcloud:
//      a. Preprocessor::preprocess_subcloud → pos (1,N,3) + x (1,4,N)
//      b. 若 N > max_n: SubcloudUtils::split_oversized 拆分为 chunks
//      c. SubcloudUtils::pad_subcloud → 填充至 min_n
//      d. CudaBuffer::upload → GPU
//      e. TrInference::infer → GPU logits (1,2,N_padded)
//      f. cudaMemcpyAsync → CPU, trim_padding → (1,2,N_true)
//      g. Transpose 至 (N_true,2) + 收集 idx_part
//   5. launch_scatter_mean_kernel → GPU 合并所有子云结果
//   6. 下载 merged logits, argmax → predictions
//   7. 返回 InferenceResult {logits, predictions, latency_ms}
// ============================================================================

#include "pipeline.h"
#include <filesystem>
#include <random>
#include <cuda_fp16.h>
#include "ply_reader.h"
#include "preprocessor.h"
#include "scatter_mean.h"
#include "stats_reader.h"
#include "subcloud_utils.h"
#include "voxelizer.h"


namespace fs = std::filesystem;

// ============================================================================
// Constructor / Destructor
// ============================================================================

InferencePipeline::InferencePipeline(
    const std::string& engine_path,
    const std::string& stats_json_path,
    TrLogger& logger,
    int min_n,
    int max_n,
    float voxel_size,
    int seed)
    : min_n_(min_n)
    , max_n_(max_n)
    , voxel_size_(voxel_size)
    , seed_(seed) {
    // 1. 加载 TensorRT 引擎
    engine_ = std::make_unique<TrEngine>(engine_path, logger);

    // 2. 创建推理执行器
    inference_ = std::make_unique<TrInference>(*engine_, logger);

    // 3. 加载特征归一化统计量 (JSON)
    stats_ = StatsReader::load(stats_json_path);

    // 4. 预分配 GPU 缓冲区 (最大子云大小)
    d_pos_ = std::make_unique<CudaBuffer>(
        static_cast<size_t>(max_n_) * 3 * sizeof(float));
    d_x_ = std::make_unique<CudaBuffer>(
        static_cast<size_t>(max_n_) * 4 * sizeof(float));
    d_logits_ = std::make_unique<CudaBuffer>(
        static_cast<size_t>(max_n_) * 2 * sizeof(float));
}

InferencePipeline::~InferencePipeline() = default;

// ============================================================================
// process_file — 单文件推理
// ============================================================================
InferenceResult InferencePipeline::process_pointcloud(const PointCloud& pc) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- Step 1: 加载 PLY ----
    const int num_points = pc.num_points;
    //PointCloud pc = PlyReader::loadv2(ply_path);
    if (num_points == 0) {
        throw std::runtime_error(
            "process_file: empty point cloud: ");
    }

    // Copy coord to mutable vector (translate-to-origin step modifies it)
    std::vector<float> coord(pc.coord.begin(), pc.coord.end());
    const float* feat_data = pc.feat.data();

    // ---- Step 2: coord -= min(coord) (匹配 Python preprocess_test) ----
    float min_c[3] = { coord[0], coord[1], coord[2] };
    for (int i = 1; i < num_points; ++i) {
        for (int c = 0; c < 3; ++c) {
            if (coord[i * 3 + c] < min_c[c]) {
                min_c[c] = coord[i * 3 + c];
            }
        }
    }
    for (int i = 0; i < num_points; ++i) {
        for (int c = 0; c < 3; ++c) {
            coord[i * 3 + c] -= min_c[c];
        }
    }

    // ---- Step 3: 体素化 → 子云列表 ----
    VoxelizeResult vox = Voxelizer::voxelize(
        coord.data(), num_points, voxel_size_, seed_);

    // ---- Step 4: 处理所有子云 ----
    // 第一遍: 计算所有子云的总点数
    int total_src = 0;
    for (const auto& part : vox.idx_points) {
        total_src += static_cast<int>(part.size());
    }

    // 分配 CPU 缓冲区, 供 scatter_mean 使用
    // all_src: (total_src, 2)  row-major
    // all_idx: (total_src,)    int64_t
    std::vector<float> all_src(total_src * 2);
    std::vector<int64_t> all_idx(total_src);

    int offset = 0;  // 当前已处理的点数 (在 all_src / all_idx 中的位置)

    for (const auto& idx_part : vox.idx_points) {
        const int N = static_cast<int>(idx_part.size());

        // (a) 预处理子云: coord 归一化 + 特征归一化
        PreprocessedCloud pp = Preprocessor::preprocess_subcloud(
            coord.data(),        // 完整点云坐标 (已移至原点)
            feat_data,         // 完整点云特征
            idx_part.data(),        // 子云索引
            N,                      // 子云大小
            stats_);                // 归一化统计量

        // (b) 若 N > max_n, 拆分为多个 chunk
        // NOTE: Preprocessing (centering + normalization) is applied to the full
        // sub-cloud before splitting, so chunking does not alter per-point statistics.
        // Set max_n large enough (e.g., 30000) to keep sub-clouds intact in practice.
        ChunkResult chunks = SubcloudUtils::split_oversized(
            pp.pos.data(), pp.x.data(), N, max_n_);

        int chunk_start = 0;  // 当前 chunk 在 idx_part 中的起始位置
        const int num_chunks = static_cast<int>(chunks.chunk_sizes.size());

        for (int c = 0; c < num_chunks; ++c) {
            const int chunk_N = chunks.chunk_sizes[c];
            const float* chunk_pos = chunks.pos_chunks[c].data();
            const float* chunk_x = chunks.x_chunks[c].data();

            // (c) 填充至 min_n
            PaddedCloud padded = SubcloudUtils::pad_subcloud(
                chunk_pos, chunk_x, chunk_N, min_n_);

            // (d) 上传至 GPU
            d_pos_->upload(
                padded.pos.data(),
                static_cast<size_t>(padded.N_padded) * 3 * sizeof(float),
                stream_.native());
            d_x_->upload(
                padded.x.data(),
                static_cast<size_t>(padded.N_padded) * 4 * sizeof(float),
                stream_.native());

            // (e) TRT 推理 → GPU logits (1, 2, N_padded)
            void* d_output = inference_->infer(
                static_cast<float*>(d_pos_->data()),
                static_cast<float*>(d_x_->data()),
                padded.N_padded,
                stream_.native());

            // (f) 下载 logits 到 CPU, 然后 trim padding
            const size_t elem_count = static_cast<size_t>(2) * padded.N_padded;
            std::vector<float> h_logits(elem_count);

            if (inference_->is_output_fp16()) {
                std::vector<__half> h_logits_half(elem_count);
                CHECK_CUDA(cudaMemcpyAsync(h_logits_half.data(), d_output, elem_count * sizeof(__half), cudaMemcpyDeviceToHost, stream_.native()));
                stream_.synchronize();
                for (size_t i = 0; i < elem_count; ++i) {
                    h_logits[i] = __half2float(h_logits_half[i]);
                }
            }
            else {
                CHECK_CUDA(cudaMemcpyAsync(h_logits.data(), d_output, elem_count * sizeof(float), cudaMemcpyDeviceToHost, stream_.native()));
                stream_.synchronize();
            }

            // Trim: 移除填充部分
            SubcloudUtils::trim_padding(
                h_logits.data(), chunk_N, padded.N_padded);

            // (g) 转置: (1, 2, chunk_N) → (chunk_N, 2) 并收集
            // h_logits 布局 (trim 后):
            //   [ch0_0 ... ch0_{N-1}, ch1_0 ... ch1_{N-1}]
            for (int i = 0; i < chunk_N; ++i) {
                all_src[(offset + i) * 2 + 0] = h_logits[i];
                all_src[(offset + i) * 2 + 1] = h_logits[chunk_N + i];
                all_idx[offset + i] = static_cast<int64_t>(
                    idx_part[chunk_start + i]);
            }

            offset += chunk_N;
            chunk_start += chunk_N;
        }  // end for chunks
    }  // end for subclouds

    // ---- Step 5: GPU scatter_mean 合并 ----
    const int N_orig = num_points;

    CudaBuffer d_src(
        static_cast<size_t>(total_src) * 2 * sizeof(float));
    CudaBuffer d_idx(
        static_cast<size_t>(total_src) * sizeof(int64_t));
    CudaBuffer d_out(
        static_cast<size_t>(N_orig) * 2 * sizeof(float));
    CudaBuffer d_cnt(
        static_cast<size_t>(N_orig) * sizeof(int));

    d_src.upload(
        all_src.data(),
        static_cast<size_t>(total_src) * 2 * sizeof(float),
        stream_.native());
    d_idx.upload(
        all_idx.data(),
        static_cast<size_t>(total_src) * sizeof(int64_t),
        stream_.native());
    launch_scatter_mean_kernel(
        static_cast<const float*>(d_src.data()),
        static_cast<const int64_t*>(d_idx.data()),
        static_cast<float*>(d_out.data()),
        static_cast<int*>(d_cnt.data()),
        total_src,
        N_orig,
        2,
        stream_.native());

    stream_.synchronize();

    // ---- Step 6: 下载合并后的 logits ----
    InferenceResult result;
    result.logits.resize(static_cast<size_t>(N_orig) * 2);
    CHECK_CUDA(cudaMemcpyAsync(
        result.logits.data(),
        d_out.data(),
        static_cast<size_t>(N_orig) * 2 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_.native()));
    stream_.synchronize();

    // ---- Step 7: Argmax (二分类: 0 vs 1) ----
    result.predictions.resize(N_orig);
    for (int i = 0; i < N_orig; ++i) {
        result.predictions[i] =
            (result.logits[i * 2 + 1] > result.logits[i * 2 + 0]) ? 1 : 0;
    }

    // ---- Step 8: 计时 ----
    auto t1 = std::chrono::high_resolution_clock::now();
    result.latency_ms =
        std::chrono::duration<float, std::milli>(t1 - t0).count();

    {
        auto mem = get_gpu_memory_info();
        std::cout << "  [GPU] " << mem.used_bytes / (1024 * 1024)
            << " MiB used for" << std::endl;
    }

    return result;
}


// ============================================================================
// process_file — 单文件推理
// ============================================================================
InferenceResult InferencePipeline::process_file(const std::string& ply_path) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- Step 1: 加载 PLY ----
    PointCloud pc = PlyReader::loadv2(ply_path);
    if (pc.num_points == 0) {
        throw std::runtime_error(
            "process_file: empty point cloud: " + ply_path);
    }

    // ---- Step 2: coord -= min(coord) (匹配 Python preprocess_test) ----
    float min_c[3] = {pc.coord[0], pc.coord[1], pc.coord[2]};
    for (int i = 1; i < pc.num_points; ++i) {
        for (int c = 0; c < 3; ++c) {
            if (pc.coord[i * 3 + c] < min_c[c]) {
                min_c[c] = pc.coord[i * 3 + c];
            }
        }
    }
    for (int i = 0; i < pc.num_points; ++i) {
        for (int c = 0; c < 3; ++c) {
            pc.coord[i * 3 + c] -= min_c[c];
        }
    }

    // ---- Step 3: 体素化 → 子云列表 ----
    VoxelizeResult vox = Voxelizer::voxelize(
        pc.coord.data(), pc.num_points, voxel_size_, seed_);

    // ---- Step 4: 处理所有子云 ----
    // 第一遍: 计算所有子云的总点数
    int total_src = 0;
    for (const auto& part : vox.idx_points) {
        total_src += static_cast<int>(part.size());
    }

    // 分配 CPU 缓冲区, 供 scatter_mean 使用
    // all_src: (total_src, 2)  row-major
    // all_idx: (total_src,)    int64_t
    std::vector<float> all_src(total_src * 2);
    std::vector<int64_t> all_idx(total_src);

    int offset = 0;  // 当前已处理的点数 (在 all_src / all_idx 中的位置)

    for (const auto& idx_part : vox.idx_points) {
        const int N = static_cast<int>(idx_part.size());

        // (a) 预处理子云: coord 归一化 + 特征归一化
        PreprocessedCloud pp = Preprocessor::preprocess_subcloud(
            pc.coord.data(),        // 完整点云坐标 (已移至原点)
            pc.feat.data(),         // 完整点云特征
            idx_part.data(),        // 子云索引
            N,                      // 子云大小
            stats_);                // 归一化统计量

        // (b) 若 N > max_n, 拆分为多个 chunk
        // NOTE: Preprocessing (centering + normalization) is applied to the full
        // sub-cloud before splitting, so chunking does not alter per-point statistics.
        // Set max_n large enough (e.g., 30000) to keep sub-clouds intact in practice.
        ChunkResult chunks = SubcloudUtils::split_oversized(
            pp.pos.data(), pp.x.data(), N, max_n_);

        int chunk_start = 0;  // 当前 chunk 在 idx_part 中的起始位置
        const int num_chunks = static_cast<int>(chunks.chunk_sizes.size());

        for (int c = 0; c < num_chunks; ++c) {
            const int chunk_N = chunks.chunk_sizes[c];
            const float* chunk_pos = chunks.pos_chunks[c].data();
            const float* chunk_x   = chunks.x_chunks[c].data();

            // (c) 填充至 min_n
            PaddedCloud padded = SubcloudUtils::pad_subcloud(
                chunk_pos, chunk_x, chunk_N, min_n_);

            // (d) 上传至 GPU
            d_pos_->upload(
                padded.pos.data(),
                static_cast<size_t>(padded.N_padded) * 3 * sizeof(float),
                stream_.native());
            d_x_->upload(
                padded.x.data(),
                static_cast<size_t>(padded.N_padded) * 4 * sizeof(float),
                stream_.native());

            // (e) TRT 推理 → GPU logits (1, 2, N_padded)
            void* d_output = inference_->infer(
                static_cast<float*>(d_pos_->data()),
                static_cast<float*>(d_x_->data()),
                padded.N_padded,
                stream_.native());

            // (f) 下载 logits 到 CPU, 然后 trim padding
            const size_t elem_count = static_cast<size_t>(2) * padded.N_padded;
            std::vector<float> h_logits(elem_count);

            if (inference_->is_output_fp16()) {
                std::vector<__half> h_logits_half(elem_count);
                CHECK_CUDA(cudaMemcpyAsync(h_logits_half.data(), d_output, elem_count * sizeof(__half), cudaMemcpyDeviceToHost, stream_.native()));
                stream_.synchronize();
                for (size_t i = 0; i < elem_count; ++i) {
                    h_logits[i] = __half2float(h_logits_half[i]);
                }
            } else {
                CHECK_CUDA(cudaMemcpyAsync(h_logits.data(), d_output, elem_count * sizeof(float), cudaMemcpyDeviceToHost, stream_.native()));
                stream_.synchronize();
            }

            // Trim: 移除填充部分
            SubcloudUtils::trim_padding(
                h_logits.data(), chunk_N, padded.N_padded);

            // (g) 转置: (1, 2, chunk_N) → (chunk_N, 2) 并收集
            // h_logits 布局 (trim 后):
            //   [ch0_0 ... ch0_{N-1}, ch1_0 ... ch1_{N-1}]
            for (int i = 0; i < chunk_N; ++i) {
                all_src[(offset + i) * 2 + 0] = h_logits[i];
                all_src[(offset + i) * 2 + 1] = h_logits[chunk_N + i];
                all_idx[offset + i] = static_cast<int64_t>(
                    idx_part[chunk_start + i]);
            }

            offset += chunk_N;
            chunk_start += chunk_N;
        }  // end for chunks
    }  // end for subclouds

    // ---- Step 5: GPU scatter_mean 合并 ----
    const int N_orig = pc.num_points;

    CudaBuffer d_src(
        static_cast<size_t>(total_src) * 2 * sizeof(float));
    CudaBuffer d_idx(
        static_cast<size_t>(total_src) * sizeof(int64_t));
    CudaBuffer d_out(
        static_cast<size_t>(N_orig) * 2 * sizeof(float));
    CudaBuffer d_cnt(
        static_cast<size_t>(N_orig) * sizeof(int));

    d_src.upload(
        all_src.data(),
        static_cast<size_t>(total_src) * 2 * sizeof(float),
        stream_.native());
    d_idx.upload(
        all_idx.data(),
        static_cast<size_t>(total_src) * sizeof(int64_t),
        stream_.native());
    launch_scatter_mean_kernel(
        static_cast<const float*>(d_src.data()),
        static_cast<const int64_t*>(d_idx.data()),
        static_cast<float*>(d_out.data()),
        static_cast<int*>(d_cnt.data()),
        total_src,
        N_orig,
        2,
        stream_.native());

    stream_.synchronize();

    // ---- Step 6: 下载合并后的 logits ----
    InferenceResult result;
    result.logits.resize(static_cast<size_t>(N_orig) * 2);
    CHECK_CUDA(cudaMemcpyAsync(
        result.logits.data(),
        d_out.data(),
        static_cast<size_t>(N_orig) * 2 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_.native()));
    stream_.synchronize();

    // ---- Step 7: Argmax (二分类: 0 vs 1) ----
    result.predictions.resize(N_orig);
    for (int i = 0; i < N_orig; ++i) {
        result.predictions[i] =
            (result.logits[i * 2 + 1] > result.logits[i * 2 + 0]) ? 1 : 0;
    }

    // ---- Step 8: 计时 ----
    auto t1 = std::chrono::high_resolution_clock::now();
    result.latency_ms =
        std::chrono::duration<float, std::milli>(t1 - t0).count();

    {
        auto mem = get_gpu_memory_info();
        std::cout << "  [GPU] " << mem.used_bytes / (1024 * 1024)
                  << " MiB used for" << ply_path << std::endl;
    }

    return result;
}


// ============================================================================
// process_directory — 处理目录下所有 PLY 文件
//
// 逻辑匹配 Python main():
//   1. 列出所有 .ply 文件, 排序
//   2. 用 seed_ 打乱
//   3. 取后 17% 作为测试集
//   4. 限制文件数 (num_files)
// ============================================================================

std::vector<InferenceResult> InferencePipeline::process_directory(
    const std::string& data_dir, int num_files) {
    // 列出所有 .ply 文件
    std::vector<fs::path> ply_paths;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ply") {
            ply_paths.push_back(entry.path());
        }
    }

    // 排序 → 打乱 (匹配 Python sorted() + np.random.seed(100).shuffle)
    std::sort(ply_paths.begin(), ply_paths.end());
    //std::mt19937 rng(seed_);
    //std::shuffle(ply_paths.begin(), ply_paths.end(), rng);

    // 取后 17% 作为测试集 (匹配 Python test split)
    const int n_total = static_cast<int>(ply_paths.size());
    const int test_start = static_cast<int>(n_total * 0.0f);
    int n_test = n_total - test_start;

    if (num_files > 0 && num_files < n_test) {
        n_test = num_files;
    }

    // 处理每个文件
    std::vector<InferenceResult> results;
    results.reserve(n_test);
    for (int i = 0; i < n_test; ++i) {
        const fs::path& p = ply_paths[test_start + i];
        results.push_back(process_file(p.string()));
    }

    return results;
}

// ============================================================================
// warmup — 随机数据预热
//
// 匹配 Python: warmup_pos = np.random.randn(1, min_n, 3)
//              warmup_x   = np.random.randn(1, 4, min_n)
// ============================================================================

void InferencePipeline::warmup(int num_runs) {
    // 创建随机输入 (尺寸 = min_n)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> pos(static_cast<size_t>(min_n_) * 3);
    std::vector<float> x(static_cast<size_t>(min_n_) * 4);
    for (int i = 0; i < min_n_ * 3; ++i) pos[i] = dist(rng);
    for (int i = 0; i < min_n_ * 4; ++i) x[i] = dist(rng);

    for (int r = 0; r < num_runs; ++r) {
        d_pos_->upload(
            pos.data(),
            static_cast<size_t>(min_n_) * 3 * sizeof(float),
            stream_.native());
        d_x_->upload(
            x.data(),
            static_cast<size_t>(min_n_) * 4 * sizeof(float),
            stream_.native());

        inference_->infer(
            static_cast<float*>(d_pos_->data()),
            static_cast<float*>(d_x_->data()),
            min_n_,
            stream_.native());

        stream_.synchronize();
    }
}
