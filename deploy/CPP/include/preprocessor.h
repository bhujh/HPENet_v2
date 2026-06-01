#pragma once
#include "types.h"
#include <vector>

/// 预处理后的子云数据，供 GPU 推理使用
struct PreprocessedCloud {
    std::vector<float> pos;  // (1, N, 3) row-major
    std::vector<float> x;    // (1, 4, N) row-major
    int N = 0;
};

/// 点云预处理：子云提取、坐标归一化、特征归一化
class Preprocessor {
public:
    /// 从完整点云中提取一个子云并进行预处理
    ///
    /// @param coord     完整点云坐标, shape (N_total × 3) row-major
    /// @param feat      完整点云特征, shape (N_total × 3) row-major
    /// @param idx_part  子云索引, shape (num_part)
    /// @param num_part  子云点数
    /// @param stats     归一化统计量 (feat_mean, feat_std, z_mean, z_std)
    /// @return PreprocessedCloud 包含 pos_batch (1,N,3) 和 x_batch (1,4,N)
    static PreprocessedCloud preprocess_subcloud(
        const float* coord,
        const float* feat,
        const int* idx_part,
        int num_part,
        const FeatureStats& stats);
};
