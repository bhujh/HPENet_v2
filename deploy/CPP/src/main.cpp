// ============================================================================
// HPENet V2 — TensorRT C++ Inference Main Entry
//
// CLI argument parser + inference pipeline skeleton.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <filesystem>

#include "types.h"
#include "logger.h"
#include "cuda_utils.h"
#include "pipeline.h"
#include "ply_reader.h"

// ---------------------------------------------------------------------------
//  CLIConfig
// ---------------------------------------------------------------------------
struct CLIConfig {
    std::string engine_path  = "deploy/trt_model_fp32.engine";
    std::string stats_path   = "deploy/CPP/stats.json";
    std::string data_dir     = "data/RadarClassi/radarfull/raw";
    int    num_files    = -1;
    int    min_n        = 1024;
    int    max_n        = 6000;
    float  voxel_size   = 0.1f;
    int    warmup       = 5;
    std::string output_path   = "./output";
    bool   benchmark    = false;
    int    seed         = 100;
};

// ---------------------------------------------------------------------------
//  Print help
// ---------------------------------------------------------------------------
static void print_help(const char* progname) {
    std::cout << "Usage: " << progname << " [OPTIONS]\n"
              << "HPENet V2 TensorRT C++ Inference Pipeline\n\n"
              << "Options:\n"
              << "  --engine=<path>         Path to TensorRT engine file\n"
              << "                           (default: deploy/trt_model_fp32.engine)\n"
              << "  --stats=<path>          Path to stats JSON file\n"
              << "                           (default: deploy/CPP/stats.json)\n"
              << "  --data_dir=<path>       Directory of test PLY files\n"
              << "                           (default: data/RadarClassi/radarfull/raw)\n"
              << "  --num_files=<int>       Number of test files (-1 = all)\n"
              << "                           (default: -1)\n"
              << "  --min_n=<int>           Minimum sub-cloud size (smaller padded)\n"
              << "                           (default: 1024)\n"
              << "  --max_n=<int>           Maximum sub-cloud size\n"
              << "                           (default: 6000)\n"
              << "  --voxel_size=<float>    Voxel size for preprocessing\n"
              << "                           (default: 0.1)\n"
              << "  --warmup=<int>          Number of warmup runs\n"
              << "                           (default: 5)\n"
              << "  --output=<path>         Output directory for results\n"
              << "                           (default: ./output)\n"
              << "  --benchmark             Enable benchmark mode\n"
              << "  --seed=<int>            Random seed\n"
              << "                           (default: 100)\n"
              << "  --help                  Print this help message and exit\n";
}

// ---------------------------------------------------------------------------
//  Hand-written CLI argument parser
// ---------------------------------------------------------------------------
static CLIConfig parse_args(int argc, char** argv) {
    CLIConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        }

        // --key=value  format
        auto eq_pos = arg.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = arg.substr(0, eq_pos);
            std::string val = arg.substr(eq_pos + 1);

            if (key == "--engine")         cfg.engine_path  = val;
            else if (key == "--stats")     cfg.stats_path   = val;
            else if (key == "--data_dir")  cfg.data_dir     = val;
            else if (key == "--num_files") cfg.num_files    = std::stoi(val);
            else if (key == "--min_n")     cfg.min_n        = std::stoi(val);
            else if (key == "--max_n")     cfg.max_n        = std::stoi(val);
            else if (key == "--voxel_size")cfg.voxel_size   = std::stof(val);
            else if (key == "--warmup")    cfg.warmup       = std::stoi(val);
            else if (key == "--output")    cfg.output_path  = val;
            else if (key == "--seed")      cfg.seed         = std::stoi(val);
            else if (key == "--benchmark") cfg.benchmark    = true;
            else {
                std::cerr << "Error: Unknown argument '" << key << "'\n";
                std::exit(1);
            }
            continue;
        }

        // --key value  format, or boolean flag
        if (arg == "--engine")      { cfg.engine_path  = argv[++i]; }
        else if (arg == "--stats")  { cfg.stats_path   = argv[++i]; }
        else if (arg == "--data_dir") { cfg.data_dir   = argv[++i]; }
        else if (arg == "--num_files"){ cfg.num_files  = std::stoi(argv[++i]); }
        else if (arg == "--min_n")  { cfg.min_n        = std::stoi(argv[++i]); }
        else if (arg == "--max_n")  { cfg.max_n        = std::stoi(argv[++i]); }
        else if (arg == "--voxel_size") { cfg.voxel_size = std::stof(argv[++i]); }
        else if (arg == "--warmup") { cfg.warmup       = std::stoi(argv[++i]); }
        else if (arg == "--output") { cfg.output_path  = argv[++i]; }
        else if (arg == "--seed")   { cfg.seed         = std::stoi(argv[++i]); }
        else if (arg == "--benchmark") { cfg.benchmark = true; }
        else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            std::exit(1);
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
//  Utility: format time duration
// ---------------------------------------------------------------------------
static std::string fmt_time(double seconds) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << seconds << "s";
    return oss.str();
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    auto config = parse_args(argc, argv);

    TrLogger logger;

    // ── Banner ──
    std::cout << "============================================================\n";
    std::cout << "HPENet V2 TensorRT C++ Inference\n";
    std::cout << "============================================================\n";
    std::cout << "Engine: " << config.engine_path << "\n";
    std::cout << "Stats:  " << config.stats_path << "\n";
    std::cout << "Data:   " << config.data_dir << "\n\n";

    // ── Initialize pipeline ──
    std::cout << "Initializing pipeline...\n";
    InferencePipeline pipeline(
        config.engine_path, config.stats_path, logger,
        config.min_n, config.max_n, config.voxel_size, config.seed
    );
    std::cout << "  Pipeline ready.\n";
    {
        //监测显存占用（Mb）
        auto mem = get_gpu_memory_info();
        std::cout << "  GPU Memory: " << mem.used_bytes / (1024 * 1024)
                  << " MiB used / " << mem.total_bytes / (1024 * 1024) << " MiB total\n";
    }

    // ── Warmup ──
    std::cout << "Warmup (" << config.warmup << " runs)...\n";
    pipeline.warmup(config.warmup);
    std::cout << "  Warmup done.\n";
    {
        auto mem = get_gpu_memory_info();
        std::cout << "  GPU Memory: " << mem.used_bytes / (1024 * 1024)
                  << " MiB used / " << mem.total_bytes / (1024 * 1024) << " MiB total\n";
    }

    // ── Process files ──
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  File                  Points     Acc      Time\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto mem = get_gpu_memory_info();
        std::cout << "  [Baseline] " << mem.used_bytes / (1024 * 1024)
                  << " MiB used / " << mem.total_bytes / (1024 * 1024) << " MiB total\n\n";
    }

    auto results = pipeline.process_directory(config.data_dir, config.num_files);

    std::vector<double> accuracies;
    std::vector<double> latencies;
    int idx = 0;

    // List files in directory for display names
    std::vector<std::string> all_ply;
    if (std::filesystem::exists(config.data_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(config.data_dir)) {
            if (entry.path().extension() == ".ply") {
                all_ply.push_back(entry.path().filename().string());
            }
        }
        std::sort(all_ply.begin(), all_ply.end());
        // Python test split: take last 17%
        int n_total = all_ply.size();
        int test_start = static_cast<int>(n_total * 0.83);
        std::vector<std::string> test_files(all_ply.begin() + test_start, all_ply.end());
        if (config.num_files > 0 && config.num_files < static_cast<int>(test_files.size())) {
            test_files.resize(config.num_files);
        }
        all_ply = test_files;
    }

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        double total_s = result.latency_ms / 1000.0;
        latencies.push_back(total_s);

        // Compute accuracy against ground-truth labels
        std::string fname = (i < all_ply.size()) ? all_ply[i] : ("file_" + std::to_string(i) + ".ply");
        std::string fpath = config.data_dir + "/" + fname;
        float acc = 0.0f;
        if (std::filesystem::exists(fpath)) {
            try {
                auto pc = PlyReader::load(fpath);
                int correct = 0;
                int n = std::min(static_cast<int>(result.predictions.size()), pc.num_points);
                for (int j = 0; j < n; ++j) {
                    if (result.predictions[j] == static_cast<int>(pc.label[j])) {
                        ++correct;
                    }
                }
                acc = (n > 0) ? static_cast<float>(correct) / n : 0.0f;
            } catch (...) {
                acc = 0.0f;
            }
        }
        accuracies.push_back(acc);

        std::cout << "  " << std::left << std::setw(20) << fname
                  << " " << std::right << std::setw(8) << result.predictions.size()
                  << "  " << std::fixed << std::setprecision(4) << acc
                  << "   " << fmt_time(total_s) << "\n";
        ++idx;
    }

    {
        auto mem = get_gpu_memory_info();
        std::cout << "------------------------------------------------------------\n";
        std::cout << "  [Final GPU Memory] " << mem.used_bytes / (1024 * 1024)
                  << " MiB used / " << mem.total_bytes / (1024 * 1024) << " MiB total\n";
    }

    // ── Summary ──
    std::cout << "------------------------------------------------------------\n";
    if (!accuracies.empty()) {
        double mean_acc = std::accumulate(accuracies.begin(), accuracies.end(), 0.0)
                          / accuracies.size();
        double mean_time = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                           / latencies.size();
        std::cout << "Mean accuracy: " << std::fixed << std::setprecision(4) << mean_acc << "\n";
        std::cout << "Mean time:     " << fmt_time(mean_time) << " per file\n";
        std::cout << "Files:         " << results.size() << "\n";
    }

    if (config.benchmark) {
        std::cout << "\nBenchmark mode enabled.\n";
    }

    std::cout << "\nDone!\n";
    return 0;
}
