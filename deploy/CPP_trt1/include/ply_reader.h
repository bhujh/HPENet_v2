#pragma once
#include"deployment_ai.h"
#include "tinyply.h"
#include "types.h"
#include <fstream>
//#include <cmath>
#include <stdexcept>
#include <cstring>

class DEPLOYAI_LIB_API PlyReader {
public:
    static PointCloud load(const std::string& path);

    static PointCloud loadv2(const std::string& path);

};


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


//template<> DEPLOYAI_LIB_API void write_annotated_ply<float>(const std::string& output_path,
//    const PointCloud& pc,
//    const std::vector<float>& predictions);

//template<> DEPLOYAI_LIB_API void write_annotated_ply<int>(const std::string& output_path,
//    const PointCloud& pc,
//    const std::vector<int>& predictions);