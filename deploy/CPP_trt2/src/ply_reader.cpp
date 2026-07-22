#define TINYPLY_IMPLEMENTATION
#include "ply_reader.h"
// #include "tinyply.h"
// #include "types.h"
#include <cmath>
#include <fstream>

namespace {

    // ---------------------------------------------------------------------------
    // Convert a tinyply Type enum to per-element stride in bytes
    // ---------------------------------------------------------------------------
    size_t type_stride(tinyply::Type t) {
        switch (t) {
        case tinyply::Type::INT8:    return 1;
        case tinyply::Type::UINT8:   return 1;
        case tinyply::Type::INT16:   return 2;
        case tinyply::Type::UINT16:  return 2;
        case tinyply::Type::INT32:   return 4;
        case tinyply::Type::UINT32:  return 4;
        case tinyply::Type::FLOAT32: return 4;
        case tinyply::Type::FLOAT64: return 8;
        default:
            throw std::runtime_error("PlyReader: unsupported PLY property type");
        }
    }

    // ---------------------------------------------------------------------------
    // Read a single numeric value from a tinyply buffer as float
    // ---------------------------------------------------------------------------
    float read_as_float(const uint8_t* ptr, tinyply::Type t) {
        switch (t) {
        case tinyply::Type::INT8:    return static_cast<float>(*reinterpret_cast<const int8_t*>(ptr));
        case tinyply::Type::UINT8:   return static_cast<float>(*reinterpret_cast<const uint8_t*>(ptr));
        case tinyply::Type::INT16:   return static_cast<float>(*reinterpret_cast<const int16_t*>(ptr));
        case tinyply::Type::UINT16:  return static_cast<float>(*reinterpret_cast<const uint16_t*>(ptr));
        case tinyply::Type::INT32:   return static_cast<float>(*reinterpret_cast<const int32_t*>(ptr));
        case tinyply::Type::UINT32:  return static_cast<float>(*reinterpret_cast<const uint32_t*>(ptr));
        case tinyply::Type::FLOAT32: return *reinterpret_cast<const float*>(ptr);
        case tinyply::Type::FLOAT64: return static_cast<float>(*reinterpret_cast<const double*>(ptr));
        default:
            throw std::runtime_error("PlyReader: unsupported PLY property type");
        }
    }

    // ---------------------------------------------------------------------------
    // Extract a column of float values from a tinyply PlyData buffer.
    // Handles all numeric types via read_as_float().
    // ---------------------------------------------------------------------------
    std::vector<float> extract_float_column(const std::shared_ptr<tinyply::PlyData>& data) {
        const size_t n = data->count;
        const size_t stride = type_stride(data->t);
        const uint8_t* buf = data->buffer.get();

        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) {
            float val = read_as_float(buf + i * stride, data->t);
            // Replace NaN with 0.0
            if (std::isnan(val)) val = 0.0f;
            out[i] = val;
        }
        return out;
    }

} // anonymous namespace

// ===========================================================================
// PlyReader::load — 读取 PLY 文件并返回 PointCloud。当.ply数据中出现NAN时，数据读取错乱
// ===========================================================================
PointCloud PlyReader::load(const std::string& path) {
    // ── 1. 打开文件 ──
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error("PlyReader: failed to open file: " + path);
    }

    // ── 2. 解析 PLY 头 ──
    tinyply::PlyFile file;
    if (!file.parse_header(stream)) {
        throw std::runtime_error("PlyReader: failed to parse PLY header: " + path);
    }

    // ── 3. 按名称请求字段 ──
    //     缺失字段会导致 tinyply 抛出 std::invalid_argument
    auto x_data = file.request_properties_from_element("vertex", { "x" });
    auto y_data = file.request_properties_from_element("vertex", { "y" });
    auto z_data = file.request_properties_from_element("vertex", { "z" });
    auto rcs_data = file.request_properties_from_element("vertex", { "rcs" });
    auto snr_data = file.request_properties_from_element("vertex", { "snr" });
    auto v_data = file.request_properties_from_element("vertex", { "v" });
    auto label_data = file.request_properties_from_element("vertex", { "label" });

    // ── 4. 读取数据 ──
    file.read(stream);

    const size_t n = x_data->count;   // 点数

    // ── 5. 提取各字段并统一转为 float32 ──
    std::vector<float> x_vals = extract_float_column(x_data);
    std::vector<float> y_vals = extract_float_column(y_data);
    std::vector<float> z_vals = extract_float_column(z_data);
    std::vector<float> rcs_vals = extract_float_column(rcs_data);
    std::vector<float> snr_vals = extract_float_column(snr_data);
    std::vector<float> v_vals = extract_float_column(v_data);
    std::vector<float> label_vals = extract_float_column(label_data);

    // ── 6. 组装 PointCloud ──
    PointCloud pc(static_cast<int>(n));
    // 验证 count 一致
    if (y_vals.size() != n || z_vals.size() != n ||
        rcs_vals.size() != n || snr_vals.size() != n ||
        v_vals.size() != n || label_vals.size() != n) {
        throw std::runtime_error("PlyReader: field count mismatch in " + path);
    }

    for (size_t i = 0; i < n; ++i) {
        pc.coord[i * 3 + 0] = x_vals[i];
        pc.coord[i * 3 + 1] = y_vals[i];
        pc.coord[i * 3 + 2] = z_vals[i];
        pc.feat[i * 3 + 0] = rcs_vals[i];
        pc.feat[i * 3 + 1] = snr_vals[i];
        pc.feat[i * 3 + 2] = v_vals[i];
        pc.label[i] = label_vals[i];
    }

    return pc;
}


// ===========================================================================
// PlyReader::loadv2 — 读取 PLY 文件并返回 PointCloud
// ===========================================================================
PointCloud PlyReader::loadv2(const std::string& path) {
    // ── 1. 打开文件 ──
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error("PlyReader: failed to open file: " + path);
    }

    // ── 2. 解析 PLY 头 ──
    tinyply::PlyFile ply_file;
    if (!ply_file.parse_header(stream)) {
        throw std::runtime_error("PlyReader: failed to parse PLY header: " + path);
    }

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
        if (p == "x")     ix_x = i;
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

        pc.coord[i * 3 + 0] = vals[ix_x];
        pc.coord[i * 3 + 1] = vals[ix_y];
        pc.coord[i * 3 + 2] = vals[ix_z];
        pc.feat[i * 3 + 0] = ix_rcs >= 0 ? vals[ix_rcs] : 0.0f;
        pc.feat[i * 3 + 1] = ix_snr >= 0 ? vals[ix_snr] : 0.0f;
        pc.feat[i * 3 + 2] = ix_v >= 0 ? vals[ix_v] : 0.0f;

        if (ix_label >= 0) {
            pc.label[i] = std::isnan(vals[ix_label]) ? 0.0f : vals[ix_label];
            pc.has_label = true;
        }
        else {
            pc.label[i] = 0.0f;
        }
    }

    return pc;
}


// ------------------ int 特化实现 ------------------
//template<>
//DEPLOYAI_LIB_API void write_annotated_ply<int>(const std::string& output_path,
//    const PointCloud& pc,
//    const std::vector<int>& predictions) {
//    int N = pc.num_points;
//    if (N == 0 || predictions.empty()) return;
//
//    std::vector<float> float6(N * 6);
//    for (int i = 0; i < N; i++) {
//        float6[i * 6 + 0] = pc.coord[i * 3 + 0];
//        float6[i * 6 + 1] = pc.coord[i * 3 + 1];
//        float6[i * 6 + 2] = pc.coord[i * 3 + 2];
//        float6[i * 6 + 3] = pc.feat[i * 3 + 0];
//        float6[i * 6 + 4] = pc.feat[i * 3 + 1];
//        float6[i * 6 + 5] = pc.feat[i * 3 + 2];
//    }
//
//    std::vector<float> label32(N);
//    for (int i = 0; i < N; i++)
//        label32[i] = static_cast<float>(predictions[i]);
//
//    std::filebuf fb;
//    if (!fb.open(output_path, std::ios::out))
//        throw std::runtime_error("Failed to open output file: " + output_path);
//    std::ostream os(&fb);
//
//    tinyply::PlyFile ply;
//    ply.add_properties_to_element("vertex", { "x","y","z","rcs","snr","v" },
//        tinyply::Type::FLOAT32, N, reinterpret_cast<const uint8_t*>(float6.data()),
//        tinyply::Type::INVALID, 0);
//    ply.add_properties_to_element("vertex", { "label" },
//        tinyply::Type::FLOAT32, N, reinterpret_cast<const uint8_t*>(label32.data()),
//        tinyply::Type::INVALID, 0);
//    ply.write(os, false);
//
//    fb.close();
//}


// ------------------ float 特化实现 ------------------
//template<>
//DEPLOYAI_LIB_API void write_annotated_ply<float>(const std::string& output_path,
//    const PointCloud& pc,
//    const std::vector<float>& predictions) {
//    // 可添加针对 float 的优化，例如直接使用 predictions 的数据（无需转换？）
//    // 但写入 PLY 需要 float_t（通常也是 float），所以可直接复制
//    // 为演示，我们调用一个内部辅助函数，或者直接复制代码
//    // 这里同样复制通用实现（略，与上面相同）
//    // 注意：如果特化内容与通用完全相同，可以完全去掉特化，但用户要求保留特化，所以提供。
//    // 实际项目中，如果确实没有差异，可以省略特化，只保留通用模板。
//    // 为了展示，我们再次实现相同逻辑（建议提取公共函数）
//    int N = pc.num_points;
//    if (N == 0 || predictions.empty()) return;
//
//    std::vector<float> float6(N * 6);
//    for (int i = 0; i < N; i++) {
//        float6[i * 6 + 0] = pc.coord[i * 3 + 0];
//        float6[i * 6 + 1] = pc.coord[i * 3 + 1];
//        float6[i * 6 + 2] = pc.coord[i * 3 + 2];
//        float6[i * 6 + 3] = pc.feat[i * 3 + 0];
//        float6[i * 6 + 4] = pc.feat[i * 3 + 1];
//        float6[i * 6 + 5] = pc.feat[i * 3 + 2];
//    }
//    std::vector<float> label32(N);
//    for (int i = 0; i < N; i++)
//        label32[i] = static_cast<float>(predictions[i]); // float -> float_t
//
//    std::filebuf fb;
//    if (!fb.open(output_path, std::ios::out))
//        throw std::runtime_error("Failed to open output file: " + output_path);
//    std::ostream os(&fb);
//    tinyply::PlyFile ply;
//    ply.add_properties_to_element("vertex", { "x","y","z","rcs","snr","v" },
//        tinyply::Type::FLOAT32, N, reinterpret_cast<const uint8_t*>(float6.data()),
//        tinyply::Type::INVALID, 0);
//    ply.add_properties_to_element("vertex", { "label" },
//        tinyply::Type::FLOAT32, N, reinterpret_cast<const uint8_t*>(label32.data()),
//        tinyply::Type::INVALID, 0);
//    ply.write(os, false);
//    fb.close();
//}