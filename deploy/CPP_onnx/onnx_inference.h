#pragma once

#include <memory>
#include <string>
#include <vector>

// ── Data structures ──────────────────────────────────────────────────────

struct PointCloud {
    std::vector<float> coord;
    std::vector<float> feat;
    std::vector<float> label;
    int num_points = 0;
    PointCloud() = default;
    PointCloud(int n) : coord(n*3), feat(n*3), label(n), num_points(n) {}
};

struct FeatureStats {
    float feat_mean[3] = {0, 0, 0};
    float feat_std[3]  = {1, 1, 1};
    float z_mean = 0;
    float z_std  = 1;
};

struct InferenceResult {
    std::string filename;
    std::vector<float> logits;
    std::vector<int>   predictions;
    float latency_ms = 0;
};

// ── Free function declarations ──────────────────────────────────────────

PointCloud load_data_ply(const std::string& data_path);
FeatureStats load_stats(const std::string& json_path);
std::vector<std::vector<int>> voxelize_cpu(const float* coord, int num_points,
                                           float voxel_size, int seed = 100);
void preprocess_subcloud(const float* coord, const float* feat,
                         const int* idx_part, int num_part,
                         const FeatureStats& stats,
                         std::vector<float>& pos_out, std::vector<float>& x_out);
std::vector<float> scatter_mean(const float* logits, const int* indices,
                                int total_points, int num_orig, int num_classes);

// ── ONNX Inference Pipeline ─────────────────────────────────────────────

class OnnxInferencePipeline {
public:
    OnnxInferencePipeline(
        const std::string& onnx_path,
        const std::string& stats_json_path,
        int min_n = 1024,
        int max_n = 30000,
        float voxel_size = 0.1f,
        int seed = 100);
    ~OnnxInferencePipeline();
    OnnxInferencePipeline(const OnnxInferencePipeline&) = delete;
    OnnxInferencePipeline& operator=(const OnnxInferencePipeline&) = delete;
    InferenceResult process_file(const std::string& ply_path);
    std::vector<InferenceResult> process_directory(
        const std::string& data_dir, int num_files = -1);
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    int min_n_, max_n_;
    float voxel_size_;
    int seed_;
};
