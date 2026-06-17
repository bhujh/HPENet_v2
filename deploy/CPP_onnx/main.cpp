#include "onnx_inference.h"

#include <cstring>
#include <iostream>

static const char* get_arg(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i < argc - 1; i++)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
static int get_int_arg(int argc, char** argv, const char* key, int def) {
    const char* v = get_arg(argc, argv, key, nullptr);
    return v ? std::atoi(v) : def;
}

int main(int argc, char** argv) {
  const char *onnx_path =
      get_arg(argc, argv, "--onnx", "../../deploy/onnx_model.onnx");
  const char *data_dir =
      get_arg(argc, argv, "--data_dir", "../../data/RadarClassi/radarfull/raw");
  const char *stats_file = get_arg(argc, argv, "--stats_file", "stats.json");
  int num_files = get_int_arg(argc, argv, "--num_files", 3);

  std::cout << "=== HPENet V2 ONNX Inference (C++) ===" << "\n"
            << "ONNX:      " << onnx_path << "\n"
            << "Data dir:  " << data_dir << "\n"
            << "Stats:     " << stats_file << "\n"
            << "Num files: " << num_files << "\n"
            << std::endl;

  try {
    OnnxInferencePipeline pipeline(onnx_path, stats_file);
    auto results = pipeline.process_directory(data_dir, num_files);

    for (size_t i = 0; i < results.size(); i++) {
      PointCloud pc = load_data_ply(results[i].filename);
      int correct = 0;
      for (int j = 0; j < pc.num_points; j++)
        if (results[i].predictions[j] == static_cast<int>(pc.label[j]))
          correct++;
      float acc = static_cast<float>(correct) / pc.num_points;

      std::cout << "[" << i << "] "
                << results[i].filename.substr(results[i].filename.rfind('/') +
                                              1)
                << "  acc=" << acc << "  latency=" << results[i].latency_ms
                << "ms"
                << "  pts=" << pc.num_points << std::endl;
    }
    std::cout << "\nDone!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
