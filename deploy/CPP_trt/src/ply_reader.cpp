#define TINYPLY_IMPLEMENTATION
#include "ply_reader.h"
#include "tinyply/tinyply.h"
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <cstring>

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
    const size_t n     = data->count;
    const size_t stride = type_stride(data->t);
    const uint8_t* buf  = data->buffer.get();

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
// PlyReader::load — 读取 PLY 文件并返回 PointCloud
// ===========================================================================
PointCloud PlyReader::load(const std::string& path) {
    // ── 1. 打开文件 ──
    std::ifstream stream(path, std::ios::binary);
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
    auto x_data     = file.request_properties_from_element("vertex", {"x"});
    auto y_data     = file.request_properties_from_element("vertex", {"y"});
    auto z_data     = file.request_properties_from_element("vertex", {"z"});
    auto rcs_data   = file.request_properties_from_element("vertex", {"rcs"});
    auto snr_data   = file.request_properties_from_element("vertex", {"snr"});
    auto v_data     = file.request_properties_from_element("vertex", {"v"});
    auto label_data = file.request_properties_from_element("vertex", {"label"});

    // ── 4. 读取数据 ──
    file.read(stream);

    const size_t n = x_data->count;   // 点数

    // ── 5. 提取各字段并统一转为 float32 ──
    std::vector<float> x_vals     = extract_float_column(x_data);
    std::vector<float> y_vals     = extract_float_column(y_data);
    std::vector<float> z_vals     = extract_float_column(z_data);
    std::vector<float> rcs_vals   = extract_float_column(rcs_data);
    std::vector<float> snr_vals   = extract_float_column(snr_data);
    std::vector<float> v_vals     = extract_float_column(v_data);
    std::vector<float> label_vals = extract_float_column(label_data);

    // ── 6. 组装 PointCloud ──
    PointCloud pc(static_cast<int>(n));
    // 验证 count 一致
    if (y_vals.size() != n || z_vals.size()   != n ||
        rcs_vals.size()  != n || snr_vals.size() != n ||
        v_vals.size()    != n || label_vals.size() != n) {
        throw std::runtime_error("PlyReader: field count mismatch in " + path);
    }

    for (size_t i = 0; i < n; ++i) {
        pc.coord[i * 3 + 0] = x_vals[i];
        pc.coord[i * 3 + 1] = y_vals[i];
        pc.coord[i * 3 + 2] = z_vals[i];
        pc.feat[i * 3 + 0]  = rcs_vals[i];
        pc.feat[i * 3 + 1]  = snr_vals[i];
        pc.feat[i * 3 + 2]  = v_vals[i];
        pc.label[i]         = label_vals[i];
    }

    return pc;
}
