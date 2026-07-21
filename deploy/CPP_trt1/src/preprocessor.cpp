#include "preprocessor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

PreprocessedCloud Preprocessor::preprocess_subcloud(
    const float* coord,
    const float* feat,
    const int* idx_part,
    int num_part,
    const FeatureStats& stats)
{
    const int N = num_part;
    const int gravity_dim = 2;  // z-axis
    PreprocessedCloud result;
    result.N = N;

    if (N <= 0) {
        return result;
    }

    // ---- step 1: gather subcloud ----
    std::vector<float> coord_part(N * 3);
    std::vector<float> feat_part(N * 3);
    for (int i = 0; i < N; i++) {
        int idx = idx_part[i];
        coord_part[i * 3 + 0] = coord[idx * 3 + 0];
        coord_part[i * 3 + 1] = coord[idx * 3 + 1];
        coord_part[i * 3 + 2] = coord[idx * 3 + 2];
        feat_part[i * 3 + 0] = feat[idx * 3 + 0];
        feat_part[i * 3 + 1] = feat[idx * 3 + 1];
        feat_part[i * 3 + 2] = feat[idx * 3 + 2];
    }

    // ---- step 2: translate to origin (subtract column-wise min) ----
    float min_coord[3] = {
        coord_part[0], coord_part[1], coord_part[2]
    };
    for (int i = 1; i < N; i++) {
        float* row = &coord_part[i * 3];
        if (row[0] < min_coord[0]) min_coord[0] = row[0];
        if (row[1] < min_coord[1]) min_coord[1] = row[1];
        if (row[2] < min_coord[2]) min_coord[2] = row[2];
    }
    for (int i = 0; i < N; i++) {
        float* row = &coord_part[i * 3];
        row[0] -= min_coord[0];
        row[1] -= min_coord[1];
        row[2] -= min_coord[2];
    }

    // ---- step 3: center positions (subtract column-wise mean) ----
    float mean_pos[3] = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < N; i++) {
        float* row = &coord_part[i * 3];
        mean_pos[0] += row[0];
        mean_pos[1] += row[1];
        mean_pos[2] += row[2];
    }
    float inv_N = 1.0f / static_cast<float>(N);
    mean_pos[0] *= inv_N;
    mean_pos[1] *= inv_N;
    mean_pos[2] *= inv_N;

    // pos = coord_part - mean_pos  (this is the "pos" tensor)
    std::vector<float> pos(N * 3);
    for (int i = 0; i < N; i++) {
        float* src = &coord_part[i * 3];
        float* dst = &pos[i * 3];
        dst[0] = src[0] - mean_pos[0];
        dst[1] = src[1] - mean_pos[1];
        dst[2] = src[2] - mean_pos[2];
    }

    // ---- step 4: zero the gravity dimension ----
    // pos[:, gravity_dim] -= pos[:, gravity_dim].min()
    float min_z = pos[0 * 3 + gravity_dim];
    for (int i = 1; i < N; i++) {
        float z = pos[i * 3 + gravity_dim];
        if (z < min_z) min_z = z;
    }
    for (int i = 0; i < N; i++) {
        pos[i * 3 + gravity_dim] -= min_z;
    }

    // ---- step 5: normalize features ----
    // feat = (feat - feat_mean) / clamp(feat_std, 1e-5)
    float feat_std_clamped[3];
    for (int c = 0; c < 3; c++) {
        feat_std_clamped[c] = (stats.feat_std[c] < 1e-5f)
            ? 1e-5f
            : stats.feat_std[c];
    }
    std::vector<float> feat_norm(N * 3);
    for (int i = 0; i < N; i++) {
        float* src = &feat_part[i * 3];
        float* dst = &feat_norm[i * 3];
        dst[0] = (src[0] - stats.feat_mean[0]) / feat_std_clamped[0];
        dst[1] = (src[1] - stats.feat_mean[1]) / feat_std_clamped[1];
        dst[2] = (src[2] - stats.feat_mean[2]) / feat_std_clamped[2];
    }

    // ---- step 6: normalize heights ----
    // heights_t = pos_t[:, gravity_dim:gravity_dim+1]  (N, 1)
    // heights = (heights - z_mean) / clamp(z_std, 1e-5)
    float z_std_clamped = (stats.z_std < 1e-5f) ? 1e-5f : stats.z_std;
    std::vector<float> height_norm(N);
    for (int i = 0; i < N; i++) {
        float z = pos[i * 3 + gravity_dim];
        height_norm[i] = (z - stats.z_mean) / z_std_clamped;
    }

    // ---- step 7: build pos_batch (1, N, 3) row-major ----
    result.pos.resize(N * 3);
    std::memcpy(result.pos.data(), pos.data(), N * 3 * sizeof(float));

    // ---- step 8: build x_batch (1, 4, N) row-major ----
    // Python: x_combined = cat([feat_norm, height_norm], dim=-1)  -> (N, 4)
    // Then: unsqueeze(0).transpose(1,2) -> (1, 4, N)
    // Memory layout: x_batch[c][n] = index c*N + n
    result.x.resize(4 * N);
    for (int i = 0; i < N; i++) {
        // feature channels (c=0,1,2)
        result.x[0 * N + i] = feat_norm[i * 3 + 0];
        result.x[1 * N + i] = feat_norm[i * 3 + 1];
        result.x[2 * N + i] = feat_norm[i * 3 + 2];
        // height channel (c=3)
        result.x[3 * N + i] = height_norm[i];
    }

    return result;
}
