#include "onnx_inference.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

static const char* get_arg(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i < argc - 1; i++)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
static bool has_arg(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --data_dir <dir> --output_dir <dir> --onnx <model.onnx> [--stats_file stats.json]\n";
}

int main(int argc, char** argv) {
    if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    const char* data_dir   = get_arg(argc, argv, "--data_dir", nullptr);
    const char* output_dir = get_arg(argc, argv, "--output_dir", nullptr);
    const char* onnx_path  = get_arg(argc, argv, "--onnx", nullptr);
    const char* stats_file = get_arg(argc, argv, "--stats_file", "stats.json");

    if (!data_dir || !output_dir || !onnx_path) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== HPENet V2 ONNX Inference (C++) ===\n"
              << "ONNX:       " << onnx_path << "\n"
              << "Data dir:   " << data_dir << "\n"
              << "Output dir: " << output_dir << "\n"
              << "Stats:      " << stats_file << "\n"
              << std::endl;

    try {
        std::filesystem::create_directories(output_dir);
        std::vector<std::string> files;
        for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
            if (entry.path().extension() == ".ply")
                files.push_back(entry.path().string());
        }
        std::sort(files.begin(), files.end());
        std::cout << "Found " << files.size() << " PLY files\n" << std::endl;

        OnnxInferencePipeline pipeline(onnx_path, stats_file);
        int total = static_cast<int>(files.size());
        double sum_acc = 0.0;
        int count_acc = 0;

        for (int i = 0; i < total; i++) {
            const auto& fpath = files[i];
            std::string fname = std::filesystem::path(fpath).filename().string();

            PointCloud pc = load_data_ply(fpath);
            auto result = pipeline.process_file(fpath);
            write_annotated_ply(std::string(output_dir) + "/" + fname, pc, result.predictions);

            if (pc.has_label) {
                int correct = 0;
                for (int j = 0; j < pc.num_points; j++)
                    if (static_cast<int>(pc.label[j]) == result.predictions[j]) correct++;
                float acc = static_cast<float>(correct) / pc.num_points;
                sum_acc += acc;
                count_acc++;
                std::cout << "[" << i << "/" << total << "] " << fname
                          << "  acc=" << acc
                          << "  latency=" << result.latency_ms << "ms"
                          << "  pts=" << pc.num_points << std::endl;
            } else {
                std::cout << "[" << i << "/" << total << "] " << fname
                          << "  acc=N/A (no labels)"
                          << "  latency=" << result.latency_ms << "ms"
                          << "  pts=" << pc.num_points << std::endl;
            }
        }

        if (count_acc > 0)
            std::cout << "\navg_acc=" << (sum_acc / count_acc)
                      << "  (" << count_acc << " files with labels)\n";
        std::cout << "\nDone!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
