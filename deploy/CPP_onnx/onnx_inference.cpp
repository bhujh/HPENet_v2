#include "onnx_inference.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply/tinyply.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// ============================================================================
//  Simple JSON parser for stats files
// ============================================================================

static std::vector<float> parse_float_array(const std::string& s) {
    std::vector<float> out;
    std::string num;
    for (char c : s) {
        if (c == '[' || c == ']' || c == ',') {
            if (!num.empty()) { out.push_back(std::stof(num)); num.clear(); }
            continue;
        }
        num += c;
    }
    if (!num.empty()) out.push_back(std::stof(num));
    return out;
}

FeatureStats load_stats(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + json_path);
    std::stringstream buf; buf << f.rdbuf();
    std::string content = buf.str();

    FeatureStats s;
    // Extract key-value pairs
    auto extract = [&](const std::string& key) -> std::string {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) return "";
        size_t val_start = content.find_first_not_of(" \t\r\n", colon + 1);
        if (val_start == std::string::npos) return "";
        char first = content[val_start];
        if (first == '[') {
            size_t close = content.find(']', val_start);
            return (close == std::string::npos) ? "" : content.substr(val_start, close - val_start + 1);
        }
        // scalar
        size_t end = content.find_first_of(",\n\r}", val_start);
        return (end == std::string::npos) ? content.substr(val_start) : content.substr(val_start, end - val_start);
    };

    std::string fm = extract("feat_mean");
    if (!fm.empty()) { auto v = parse_float_array(fm); for (int i=0; i<3&&i<(int)v.size(); i++) s.feat_mean[i]=v[i]; }

    std::string fs = extract("feat_std");
    if (!fs.empty()) { auto v = parse_float_array(fs); for (int i=0; i<3&&i<(int)v.size(); i++) s.feat_std[i]=v[i]; }

    std::string zm = extract("z_mean");
    if (!zm.empty()) s.z_mean = std::stof(zm);

    std::string zs = extract("z_std");
    if (!zs.empty()) s.z_std = std::stof(zs);

    return s;
}

// ============================================================================
//  PLY reader
// ============================================================================

PointCloud load_data_ply(const std::string& data_path) {
    // Phase 1: parse header with tinyply (robust, handles all edge cases)
    std::ifstream stream(data_path, std::ios::binary);
    if (!stream) throw std::runtime_error("Failed to open: " + data_path);

    tinyply::PlyFile ply_file;
    ply_file.parse_header(stream);

    int num_vertices = 0;
    std::vector<std::string> prop_names;
    bool has_label_field = false;

    for (const auto& elem : ply_file.get_elements()) {
        if (elem.name == "vertex") {
            num_vertices = static_cast<int>(elem.size);
            for (const auto& prop : elem.properties) {
                prop_names.push_back(prop.name);
                if (prop.name == "label") has_label_field = true;
            }
            break;
        }
    }

    if (num_vertices <= 0) throw std::runtime_error("No vertex element in PLY");

    // Map property names → column indices
    int ix_x = -1, ix_y = -1, ix_z = -1,
        ix_rcs = -1, ix_snr = -1, ix_v = -1, ix_label = -1;
    for (int i = 0; i < (int)prop_names.size(); i++) {
        const auto& p = prop_names[i];
        if      (p == "x")     ix_x = i;
        else if (p == "y")     ix_y = i;
        else if (p == "z")     ix_z = i;
        else if (p == "rcs")   ix_rcs = i;
        else if (p == "snr")   ix_snr = i;
        else if (p == "v")     ix_v = i;
        else if (p == "label") ix_label = i;
    }

    if (ix_x < 0 || ix_y < 0 || ix_z < 0)
        throw std::runtime_error("PLY missing x/y/z field");

    // Phase 2: read data ourselves (tinyply's read() can't handle 'nan' literals)
    stream.clear(); // in case parse_header set eof/fail
    PointCloud pc(num_vertices);
    int nprop = (int)prop_names.size();
    std::vector<float> vals(nprop);
    std::string line;

    for (int i = 0; i < num_vertices; i++) {
        if (!std::getline(stream, line))
            throw std::runtime_error(
                "PLY truncated: expected " + std::to_string(num_vertices)
                + " points, got " + std::to_string(i));
        std::istringstream iss(line);
        for (int j = 0; j < nprop; j++)
            if (!(iss >> vals[j])) vals[j] = 0.0f;
        for (int j = 0; j < nprop; j++)
            if (std::isnan(vals[j])) vals[j] = 0.0f;

        pc.coord[i*3+0] = vals[ix_x];
        pc.coord[i*3+1] = vals[ix_y];
        pc.coord[i*3+2] = vals[ix_z];
        pc.feat[i*3+0]  = ix_rcs >= 0 ? vals[ix_rcs] : 0.0f;
        pc.feat[i*3+1]  = ix_snr >= 0 ? vals[ix_snr] : 0.0f;
        pc.feat[i*3+2]  = ix_v >= 0   ? vals[ix_v]   : 0.0f;

        if (ix_label >= 0) {
            pc.label[i] = std::isnan(vals[ix_label]) ? 0.0f : vals[ix_label];
            pc.has_label = true;
        } else {
            pc.label[i] = 0.0f;
        }
    }

    return pc;
}

// ============================================================================
//  CPU Voxelizer (FNV-1A hash + stable sort + sub-cloud extraction)
//  Matches Python voxelize(coord, voxel_size, mode=1) from openpoints.
// ============================================================================

// FNV-1A 64-bit hash of a 3D coordinate
static uint64_t fnv_hash_coord(const int64_t* coord3) {
    uint64_t hash = 14695981039346656037ULL;
    for (int j = 0; j < 3; j++) {
        hash *= 1099511628211ULL;
        hash ^= static_cast<uint64_t>(coord3[j]);
    }
    return hash;
}

// Map int64 3D coord to a flat hash key
static int64_t coord_to_int(float c, float voxel_size) {
    return static_cast<int64_t>(std::floor(c / voxel_size));
}

static uint64_t hash_coord(float x, float y, float z, float voxel_size) {
    int64_t d[3] = {coord_to_int(x, voxel_size), coord_to_int(y, voxel_size), coord_to_int(z, voxel_size)};
    return fnv_hash_coord(d);
}

std::vector<std::vector<int>> voxelize_cpu(const float* coord, int num_points,
                                            float voxel_size, int seed) {
    // Step 1: compute hash key for each point
    struct PointHash {
        int idx;
        uint64_t hash;
    };
    std::vector<PointHash> ph(num_points);
    for (int i = 0; i < num_points; i++) {
        int i3 = i * 3;
        // Replace NaN with 0
        float x = std::isnan(coord[i3])     ? 0.0f : coord[i3];
        float y = std::isnan(coord[i3 + 1]) ? 0.0f : coord[i3 + 1];
        float z = std::isnan(coord[i3 + 2]) ? 0.0f : coord[i3 + 2];
        ph[i] = {i, hash_coord(x, y, z, voxel_size)};
    }

    // Step 2: stable sort by hash
    std::stable_sort(ph.begin(), ph.end(),
                     [](const PointHash& a, const PointHash& b) { return a.hash < b.hash; });

    // Step 3: group voxels and count points per voxel
    std::vector<int> idx_sort(num_points);
    std::vector<int> counts;
    for (int i = 0; i < num_points; ) {
        uint64_t cur_hash = ph[i].hash;
        int j = i;
        while (j < num_points && ph[j].hash == cur_hash) j++;
        int cnt = j - i;
        counts.push_back(cnt);
        for (int k = i; k < j; k++) {
            idx_sort[k] = ph[k].idx;
        }
        i = j;
    }

    int num_voxels = static_cast<int>(counts.size());
    int max_count = 0;
    for (int c : counts) max_count = std::max(max_count, c);
    if (max_count == 0) return {};

    std::vector<int> cumsum(num_voxels + 1, 0);
    for (int i = 0; i < num_voxels; i++) {
        cumsum[i + 1] = cumsum[i] + counts[i];
    }

    std::mt19937 rng(seed);
    std::vector<std::vector<int>> idx_points;

    for (int i = 0; i < max_count; i++) {
        std::vector<int> part;
        part.reserve(num_voxels);
        for (int v = 0; v < num_voxels; v++) {
            int offset = cumsum[v] + (i % counts[v]);
            part.push_back(idx_sort[offset]);
        }
        // Shuffle the sub-cloud (matches Python np.random.shuffle)
        std::shuffle(part.begin(), part.end(), rng);
        idx_points.push_back(std::move(part));
    }

    return idx_points;
}

// ============================================================================
//  Preprocessor — XYZAlign + feature normalization
// ============================================================================

void preprocess_subcloud(const float* coord, const float* feat,
                         const int* idx_part, int num_part,
                         const FeatureStats& stats,
                         std::vector<float>& pos_out, std::vector<float>& x_out) {
    std::vector<float> coord_part(num_part * 3);
    std::vector<float> feat_part(num_part * 3);
    for (int i = 0; i < num_part; i++) {
        int src = idx_part[i] * 3;
        int dst = i * 3;
        coord_part[dst]     = coord[src];
        coord_part[dst + 1] = coord[src + 1];
        coord_part[dst + 2] = coord[src + 2];
        feat_part[dst]      = feat[src];
        feat_part[dst + 1]  = feat[src + 1];
        feat_part[dst + 2]  = feat[src + 2];
    }

    // XYZAlign: mean-center xy, z-min
    float mean_x = 0, mean_y = 0;
    float min_z  = std::numeric_limits<float>::max();
    for (int i = 0; i < num_part; i++) {
        int i3 = i * 3;
        mean_x += coord_part[i3];
        mean_y += coord_part[i3 + 1];
        if (coord_part[i3 + 2] < min_z) min_z = coord_part[i3 + 2];
    }
    mean_x /= num_part;
    mean_y /= num_part;

    pos_out.resize(num_part * 3);
    x_out.resize(num_part * 4);

    for (int i = 0; i < num_part; i++) {
        int i3 = i * 3;
        float px = coord_part[i3]     - mean_x;
        float py = coord_part[i3 + 1] - mean_y;
        float pz = coord_part[i3 + 2] - min_z;

        pos_out[i3]     = px;
        pos_out[i3 + 1] = py;
        pos_out[i3 + 2] = pz;

        float f0 = (feat_part[i3]     - stats.feat_mean[0]) / std::max(stats.feat_std[0], 1e-5f);
        float f1 = (feat_part[i3 + 1] - stats.feat_mean[1]) / std::max(stats.feat_std[1], 1e-5f);
        float f2 = (feat_part[i3 + 2] - stats.feat_mean[2]) / std::max(stats.feat_std[2], 1e-5f);

        float hn = (pz - stats.z_mean) / std::max(stats.z_std, 1e-5f);

        x_out[0 * num_part + i] = f0;
        x_out[1 * num_part + i] = f1;
        x_out[2 * num_part + i] = f2;
        x_out[3 * num_part + i] = hn;
    }
}

// ============================================================================
//  CPU Scatter Mean — matches torch_scatter.scatter(..., reduce='mean')
// ============================================================================

std::vector<float> scatter_mean(const float* logits, const int* indices,
                                int total_points, int num_orig, int num_classes) {
    std::vector<float> sum(num_orig * num_classes, 0.0f);
    std::vector<int> count(num_orig, 0);

    for (int i = 0; i < total_points; i++) {
        int idx = indices[i];
        for (int c = 0; c < num_classes; c++) {
            sum[idx * num_classes + c] += logits[i * num_classes + c];
        }
        count[idx]++;
    }

    std::vector<float> result(num_orig * num_classes, 0.0f);
    for (int i = 0; i < num_orig; i++) {
        if (count[i] > 0) {
            float inv = 1.0f / static_cast<float>(count[i]);
            for (int c = 0; c < num_classes; c++) {
                result[i * num_classes + c] = sum[i * num_classes + c] * inv;
            }
        }
    }
    return result;
}

// ============================================================================
//  ONNX Engine — PIMPL implementation
// ============================================================================

struct OnnxInferencePipeline::Impl {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> in_storage;   // owns input name strings
    std::vector<std::string> out_storage;  // owns output name strings
    FeatureStats stats;
    int min_n, max_n;
    float voxel_size;
    int seed;

    Impl(const std::string& onnx_path, const std::string& stats_path,
         int min_n_, int max_n_, float voxel_size_, int seed_)
        : env(ORT_LOGGING_LEVEL_WARNING, "hpenet_onnx"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          min_n(min_n_), max_n(max_n_), voxel_size(voxel_size_), seed(seed_)
    {
        stats = load_stats(stats_path);

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, onnx_path.c_str(), opts);

        // Query I/O names from session into persistent storage
        {
            auto alloc = Ort::AllocatorWithDefaultOptions();
            size_t num_in = session.GetInputCount();
            size_t num_out = session.GetOutputCount();
            for (size_t i = 0; i < num_in; i++) {
                auto name = session.GetInputNameAllocated(i, alloc);
                in_storage.push_back(name.get());
            }
            for (size_t i = 0; i < num_out; i++) {
                auto name = session.GetOutputNameAllocated(i, alloc);
                out_storage.push_back(name.get());
            }
        }
        for (const auto& s : in_storage) input_names.push_back(s.c_str());
        for (const auto& s : out_storage) output_names.push_back(s.c_str());
    }
};

OnnxInferencePipeline::OnnxInferencePipeline(
    const std::string& onnx_path, const std::string& stats_json_path,
    int min_n, int max_n, float voxel_size, int seed)
    : pimpl_(std::make_unique<Impl>(onnx_path, stats_json_path,
                                    min_n, max_n, voxel_size, seed)),
      min_n_(min_n), max_n_(max_n), voxel_size_(voxel_size), seed_(seed) {}

OnnxInferencePipeline::~OnnxInferencePipeline() = default;

InferenceResult OnnxInferencePipeline::process_file(const std::string& ply_path) {
    auto& impl = *pimpl_;
    auto start_time = std::chrono::steady_clock::now();

    PointCloud pc = load_data_ply(ply_path);
    int N = pc.num_points;
    if (N == 0) {
        InferenceResult empty;
        empty.filename = ply_path;
        return empty;
    }
    if (N < min_n_) {
        std::cerr << "WARNING: " << ply_path << " has " << N
                  << " points (< min_n=" << min_n_ << "), skipping\n";
        InferenceResult empty;
        empty.filename = ply_path;
        empty.predictions.resize(N, 0);
        return empty;
    }
    if (N > max_n_) {
        std::cerr << "WARNING: " << ply_path << " has " << N
                  << " points (> max_n=" << max_n_ << "), truncating\n";
    }

    // Translate to origin: coord -= min(coord)
    float mx = pc.coord[0], my = pc.coord[1], mz = pc.coord[2];
    for (int i = 0; i < N; i++) {
        if (pc.coord[i*3]     < mx) mx = pc.coord[i*3];
        if (pc.coord[i*3 + 1] < my) my = pc.coord[i*3 + 1];
        if (pc.coord[i*3 + 2] < mz) mz = pc.coord[i*3 + 2];
    }
    for (int i = 0; i < N; i++) {
        pc.coord[i*3]     -= mx;
        pc.coord[i*3 + 1] -= my;
        pc.coord[i*3 + 2] -= mz;
    }

    // Voxelize
    auto idx_points = voxelize_cpu(pc.coord.data(), N, voxel_size_, seed_);

    // Per sub-cloud: preprocess + ONNX inference
    constexpr int kMinSubcloudPoints = 16;
    std::vector<float> all_logits;
    std::vector<int> all_indices;
    int total_sub = 0;
    int skipped_sub = 0;

    for (const auto& part : idx_points) {
        int np = static_cast<int>(part.size());
        if (np == 0) continue;
        if (np < kMinSubcloudPoints) { skipped_sub++; continue; }

        std::vector<float> pos_out, x_out;
        preprocess_subcloud(pc.coord.data(), pc.feat.data(),
                            part.data(), np, impl.stats,
                            pos_out, x_out);

        // ONNX inference
        std::array<int64_t, 3> pos_shape = {1, np, 3};
        std::array<int64_t, 3> x_shape   = {1, 4, np};

        auto pos_t = Ort::Value::CreateTensor<float>(
            impl.memory_info, pos_out.data(), pos_out.size(), pos_shape.data(), 3);
        auto x_t = Ort::Value::CreateTensor<float>(
            impl.memory_info, x_out.data(), x_out.size(), x_shape.data(), 3);

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(pos_t));
        ort_inputs.push_back(std::move(x_t));

        Ort::RunOptions run_opts;
        try {
            auto outputs = impl.session.Run(run_opts, impl.input_names.data(),
                                            ort_inputs.data(), 2,
                                            impl.output_names.data(), 1);

        float* out_data = outputs[0].GetTensorMutableData<float>();
        auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
        int64_t out_n = out_info.GetShape()[2];
        int num_class = 2;

        for (int j = 0; j < out_n; j++) {
            for (int c = 0; c < num_class; c++) {
                all_logits.push_back(out_data[c * out_n + j]);
            }
            all_indices.push_back(part[j]);
            total_sub++;
        }
        } catch (const std::exception& e) {
            skipped_sub++;
            std::cerr << "WARNING: " << ply_path
                      << " sub-cloud (np=" << np << ") ONNX inference failed: "
                      << e.what() << ", skipping\n";
        }
    }

    if (skipped_sub > 0) {
        std::cerr << "WARNING: " << ply_path << " skipped " << skipped_sub
                  << " sub-cloud" << (skipped_sub > 1 ? "s" : "")
                  << " (< " << kMinSubcloudPoints << " points)\n";
    }

    if (total_sub == 0) {
        InferenceResult empty;
        empty.logits.resize(N * 2, 0.0f);
        empty.predictions.resize(N, 0);
        empty.latency_ms = 0;
        return empty;
    }

    // Scatter mean + argmax
    auto merged = scatter_mean(all_logits.data(), all_indices.data(),
                               total_sub, N, 2);

    InferenceResult result;
    result.filename = ply_path;
    result.logits = merged;
    result.predictions.resize(N);
    for (int i = 0; i < N; i++) {
        result.predictions[i] = (merged[i * 2 + 1] > merged[i * 2]) ? 1 : 0;
    }

    auto end_time = std::chrono::steady_clock::now();
    result.latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time).count() / 1000.0f;

    return result;
}

void write_annotated_ply(const std::string& output_path,
                         const PointCloud& pc,
                         const std::vector<int>& predictions) {
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

    std::vector<int32_t> label32(N);
    for (int i = 0; i < N; i++)
        label32[i] = static_cast<int32_t>(predictions[i]);

    std::filebuf fb;
    if (!fb.open(output_path, std::ios::out))
        throw std::runtime_error("Failed to open output file: " + output_path);
    std::ostream os(&fb);

    tinyply::PlyFile ply;
    ply.add_properties_to_element("vertex", {"x","y","z","rcs","snr","v"},
        tinyply::Type::FLOAT32, N, reinterpret_cast<const uint8_t*>(float6.data()),
        tinyply::Type::INVALID, 0);
    ply.add_properties_to_element("vertex", {"label"},
        tinyply::Type::INT32, N, reinterpret_cast<const uint8_t*>(label32.data()),
        tinyply::Type::INVALID, 0);
    ply.write(os, false);

    fb.close();
}
