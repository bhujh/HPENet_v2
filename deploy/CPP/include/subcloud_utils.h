#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// PaddedCloud — result of pad_subcloud
// pos: (1, padded_N, 3)  row-major
// x:   (1, 4, padded_N)  channel-major (4 channels of padded_N each)
// ---------------------------------------------------------------------------
struct PaddedCloud {
    std::vector<float> pos;  // (1, padded_N, 3)
    std::vector<float> x;    // (1, 4, padded_N)
    int N_padded = 0;
};

// ---------------------------------------------------------------------------
// ChunkResult — result of split_oversized
// Each chunk has its own pos (1, chunk_N, 3) and x (1, 4, chunk_N).
// ---------------------------------------------------------------------------
struct ChunkResult {
    std::vector<std::vector<float>> pos_chunks;
    std::vector<std::vector<float>> x_chunks;
    std::vector<int> chunk_sizes;
};

// ---------------------------------------------------------------------------
// SubcloudUtils — point cloud padding / splitting / trimming utilities
//
// These mirror the Python helpers in deploy/trt_inference.py and are used
// to prepare sub-clouds for TensorRT inference (which has a fixed min/max
// input size).
// ---------------------------------------------------------------------------
class SubcloudUtils {
public:
    /// Pad sub-cloud to exactly @p min_n points by replicating the last point.
    /// If N >= min_n the input is copied as-is.
    /// @param pos  (1, N, 3) row-major float array
    /// @param x    (1, 4, N) channel-major float array
    /// @param N    number of points
    /// @param min_n  target point count
    /// @return PaddedCloud with pos (1, padded_N, 3) and x (1, 4, padded_N)
    static PaddedCloud pad_subcloud(
        const float* pos,
        const float* x,
        int N,
        int min_n
    );

    /// Split a sub-cloud with N > max_n into multiple chunks, each with at
    /// most @p max_n points.  If N <= max_n a single chunk is returned.
    /// @param pos   (1, N, 3) row-major float array
    /// @param x     (1, 4, N) channel-major float array
    /// @param N     number of points
    /// @param max_n  max points per chunk
    /// @return ChunkResult with ceil(N/max_n) chunks
    static ChunkResult split_oversized(
        const float* pos,
        const float* x,
        int N,
        int max_n
    );

    /// Trim padded logits back to the original point count in-place.
    /// logits layout: (1, 2, padded_N) channel-major.
    /// After trimming only the first N_true columns of each channel remain.
    /// @param logits  (1, 2, padded_N) — modified in-place to (1, 2, N_true)
    /// @param N_true  number of valid points
    /// @param padded_N  padded size (must be >= N_true)
    static void trim_padding(
        float* logits,
        int N_true,
        int padded_N
    );
};

// ---------------------------------------------------------------------------
// Free-standing convenience wrappers (return new vectors rather than in-place)
// ---------------------------------------------------------------------------

/// Convenience wrapper: pads (pos, x) to min_n.  Returns PaddedCloud.
inline PaddedCloud pad_subcloud(
    const std::vector<float>& pos,
    const std::vector<float>& x,
    int N,
    int min_n)
{
    return SubcloudUtils::pad_subcloud(pos.data(), x.data(), N, min_n);
}

/// Convenience wrapper: splits oversized cloud into chunks.
inline ChunkResult split_oversized(
    const std::vector<float>& pos,
    const std::vector<float>& x,
    int N,
    int max_n)
{
    return SubcloudUtils::split_oversized(pos.data(), x.data(), N, max_n);
}
