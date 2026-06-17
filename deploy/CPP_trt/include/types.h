#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct PointCloud {
    std::vector<float> coord;  // N×3, row-major
    std::vector<float> feat;   // N×3, row-major
    std::vector<float> label;  // N
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

struct Config {
    std::string engine_path  = "deploy/trt_model_fp32.engine";
    std::string stats_path   = "deploy/stats.json";
    std::string data_dir     = "data/RadarClassi/radarfull/raw";
    int    num_files    = -1;   // -1 = all
    int    min_n        = 1024;
    int    max_n        = 30000;
    float  voxel_size   = 0.1f;
    int    warmup_runs  = 5;
    std::string output_path   = "./output";
    int    seed         = 100;
    bool   benchmark    = false;
};

struct InferenceResult {
    std::vector<float> logits;       // N×2, row-major
    std::vector<int>   predictions;  // N (argmax)
    float latency_ms = 0;
};
