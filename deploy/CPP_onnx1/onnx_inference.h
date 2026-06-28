#pragma once
// 跨平台导出宏定义
#ifdef _WIN32
#ifdef DEPLOYAI_LIB_EXPORTS
#define DEPLOYAI_LIB_API __declspec(dllexport)
#else
#define DEPLOYAI_LIB_API __declspec(dllimport)
#endif
#else
    // Linux/macOS 下默认全局符号可见，或显式指定
#define DEPLOYAI_LIB_API __attribute__((visibility("default")))
#endif

#include "tinyply.h"
#include <fstream>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

// ── Data structures ──────────────────────────────────────────────────────

struct PointCloud {
    std::vector<float> coord;
    std::vector<float> feat;
    std::vector<float> label;
    int num_points = 0;
    bool has_label = false;
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

DEPLOYAI_LIB_API PointCloud load_data_ply(const std::string& data_path);
DEPLOYAI_LIB_API FeatureStats load_stats(const std::string& json_path);
std::vector<std::vector<int>> voxelize_cpu(const float* coord, int num_points,
                                           float voxel_size, int seed = 100);
void preprocess_subcloud(const float* coord, const float* feat,
                         const int* idx_part, int num_part,
                         const FeatureStats& stats,
                         std::vector<float>& pos_out, std::vector<float>& x_out);
std::vector<float> scatter_mean(const float* logits, const int* indices,
                                int total_points, int num_orig, int num_classes);

template<typename T>
inline void write_annotated_ply(const std::string& output_path,
    const PointCloud& pc,
    const std::vector<T>& predictions) {
    int N = pc.num_points;
    if (N == 0 || predictions.empty()) return;

    std::vector<float> float6(N * 6);
    for (int i = 0; i < N; i++) {
        float6[i * 6 + 0] = pc.coord[i * 3 + 0];
        float6[i * 6 + 1] = pc.coord[i * 3 + 1];
        float6[i * 6 + 2] = pc.coord[i * 3 + 2];
        float6[i * 6 + 3] = pc.feat[i * 3 + 0];
        float6[i * 6 + 4] = pc.feat[i * 3 + 1];
        float6[i * 6 + 5] = pc.feat[i * 3 + 2];
    }

    std::vector<float> label32(N);
    for (int i = 0; i < N; i++)
        label32[i] = static_cast<float>(predictions[i]);

    std::filebuf fb;
    if (!fb.open(output_path, std::ios::out))
        throw std::runtime_error("Failed to open output file: " + output_path);
    std::ostream os(&fb);

    tinyply::PlyFile ply;
    ply.add_properties_to_element("vertex", { "x","y","z","rcs","snr","v" },
        tinyply::Type::FLOAT32, N, reinterpret_cast<const uint8_t*>(float6.data()),
        tinyply::Type::INVALID, 0);
    ply.add_properties_to_element("vertex", { "label" },
        tinyply::Type::FLOAT32, N, reinterpret_cast<const uint8_t*>(label32.data()),
        tinyply::Type::INVALID, 0);
    ply.write(os, false);

    fb.close();
}


template<> DEPLOYAI_LIB_API void write_annotated_ply<float>(const std::string& output_path,
    const PointCloud& pc,
    const std::vector<float>& predictions);

template<> DEPLOYAI_LIB_API void write_annotated_ply<int>(const std::string& output_path,
    const PointCloud& pc,
    const std::vector<int>& predictions);

// ── ONNX Inference Pipeline ─────────────────────────────────────────────

class DEPLOYAI_LIB_API OnnxInferencePipeline {
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
    InferenceResult process_pointcloud(const PointCloud& pc);
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    int min_n_, max_n_;
    float voxel_size_;
    int seed_;
};


