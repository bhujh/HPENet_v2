#pragma once
#include"deployment_ai.h"
#include "cuda_utils.h"
#include "types.h"
#include "trt_engine.h"
#include "trt_inference.h"

//#include "ply_reader.h"
//#include "preprocessor.h"
//#include "scatter_mean.h"
//#include "stats_reader.h"
//#include "subcloud_utils.h"
//#include "voxelizer.h"

//#include <memory>
//#include <string>
//#include <vector>
//#include <algorithm>
//#include <chrono>
//#include <cmath>
//#include <cstring>
//#include <fstream>
//#include <iostream>



/// InferencePipeline — 端到端推理流水线
///
/// 用法:
///   TrLogger logger;
///   InferencePipeline pipeline(
///       "deploy/trt_model_fp32.engine",
///       "deploy/stats.json",
///       logger);
///   auto result = pipeline.process_file("data/test.ply");
class DEPLOYAI_LIB_API InferencePipeline {
public:
    /// @param engine_path      TensorRT .engine 文件路径
    /// @param stats_json_path  特征归一化统计量 JSON (由 convert_stats.py 生成)
    /// @param logger           TrLogger 日志实例
    /// @param min_n            子云最小点数 (不足则填充)
    /// @param max_n            子云最大点数 (超出则拆分)
    /// @param voxel_size       体素大小 (米)
    /// @param seed             体素化随机种子
    InferencePipeline(
        const std::string& engine_path,
        const std::string& stats_json_path,
        TrLogger& logger,
        int min_n = 1024,
        int max_n = 30000,
        float voxel_size = 0.1f,
        int seed = 100);

    ~InferencePipeline();

    // 禁止拷贝
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;

    // ── 公共接口 ────────────────────────────────────────────────────────

    /// 处理单个 PLY 文件
    /// @param ply_path  .ply 文件路径
    /// @return InferenceResult 包含 logits、predictions、latency_ms
    InferenceResult process_file(const std::string& ply_path);

    /// 处理单个 PLY 文件
    /// @param ply_path  .ply 文件路径
    /// @return InferenceResult 包含 logits、predictions、latency_ms
    InferenceResult process_pointcloud(const PointCloud& pc);

    /// 处理目录下所有 PLY 文件 (按 Python test split 逻辑: 取后 17%)
    /// @param data_dir  包含 .ply 文件的目录
    /// @param num_files 处理文件数, -1 = 全部
    /// @return vector<InferenceResult> 每个文件一个结果
    std::vector<InferenceResult> process_directory(
        const std::string& data_dir,
        int num_files = -1);

    /// Warmup: 用随机数据跑 N 次推理
    /// @param num_runs  warmup 次数 (默认 5)
    void warmup(int num_runs = 5);

private:
    // ── 成员 ────────────────────────────────────────────────────────────
    std::unique_ptr<TrEngine> engine_;
    std::unique_ptr<TrInference> inference_;
    std::unique_ptr<CudaBuffer> d_pos_;      // GPU: (1, max_n, 3)
    std::unique_ptr<CudaBuffer> d_x_;        // GPU: (1, 4, max_n)
    std::unique_ptr<CudaBuffer> d_logits_;   // GPU: (1, 2, max_n)
    CudaStream stream_;
    FeatureStats stats_;
    int min_n_;
    int max_n_;
    float voxel_size_;
    int seed_;
};
