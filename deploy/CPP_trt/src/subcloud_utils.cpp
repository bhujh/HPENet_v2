#include "subcloud_utils.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

// ===========================================================================
// pad_subcloud
// ===========================================================================
// Python equivalent (trt_inference.py:40–52):
//   pos_pad = np.concatenate([pos, np.tile(pos[:, -1:, :], (1, pad, 1))], axis=1)
//   x_pad   = np.concatenate([x,   np.tile(x[:, :, -1:], (1, 1, pad))], axis=2)
// ===========================================================================
PaddedCloud SubcloudUtils::pad_subcloud(
    const float* pos,
    const float* x,
    int N,
    int min_n)
{
    if (N < 0 || min_n < 1) {
        throw std::invalid_argument(
            "pad_subcloud: N must be >= 0 and min_n >= 1");
    }

    if (N >= min_n) {
        // No padding needed — copy as-is
        PaddedCloud result;
        result.pos.assign(pos, pos + N * 3);
        result.x.assign(x, x + 4 * N);
        result.N_padded = N;
        return result;
    }

    const int pad = min_n - N;  // number of padding points

    PaddedCloud result;
    result.N_padded = min_n;

    // ── pos: (1, N, 3) → (1, min_n, 3) ──
    result.pos.resize(min_n * 3);

    // Copy original N points
    if (N > 0) {
        std::memcpy(result.pos.data(), pos, N * 3 * sizeof(float));

        // Fill remaining positions with the last point
        const float* last_point = pos + (N - 1) * 3;
        for (int i = 0; i < pad; ++i) {
            float* dst = result.pos.data() + (N + i) * 3;
            std::memcpy(dst, last_point, 3 * sizeof(float));
        }
    }
    // If N == 0, pos is entirely zeros — already zero-initialized from resize().
    // (resize() value-initialises new elements to 0.0f)

    // ── x: (1, 4, N) → (1, 4, min_n) ──
    result.x.resize(4 * min_n);

    if (N > 0) {
        for (int c = 0; c < 4; ++c) {
            // Copy original N values for this channel
            const float* src_ch = x + c * N;
            float* dst_ch = result.x.data() + c * min_n;
            std::memcpy(dst_ch, src_ch, N * sizeof(float));

            // Fill trailing pad values with the last value of this channel
            const float last_val = src_ch[N - 1];
            for (int i = 0; i < pad; ++i) {
                dst_ch[N + i] = last_val;
            }
        }
    }
    // If N == 0, result.x is already zero-initialised.

    return result;
}

// ===========================================================================
// split_oversized
// ===========================================================================
// Splits a sub-cloud with N > max_n into ceil(N / max_n) chunks.
// Each chunk has at most max_n points (the last chunk may be smaller).
// ===========================================================================
ChunkResult SubcloudUtils::split_oversized(
    const float* pos,
    const float* x,
    int N,
    int max_n)
{
    if (N < 0 || max_n < 1) {
        throw std::invalid_argument(
            "split_oversized: N must be >= 0 and max_n >= 1");
    }

    ChunkResult result;

    if (N <= max_n) {
        // Single chunk
        result.pos_chunks.emplace_back(pos, pos + N * 3);
        result.x_chunks.emplace_back(x, x + 4 * N);
        result.chunk_sizes.push_back(N);
        return result;
    }

    const int num_chunks = (N + max_n - 1) / max_n;  // ceil division

    result.pos_chunks.reserve(num_chunks);
    result.x_chunks.reserve(num_chunks);
    result.chunk_sizes.reserve(num_chunks);

    int offset = 0;
    for (int c = 0; c < num_chunks; ++c) {
        const int remaining = N - offset;
        const int chunk_n = std::min(max_n, remaining);

        result.chunk_sizes.push_back(chunk_n);

        // Copy pos for this chunk: (1, chunk_n, 3)
        std::vector<float> pos_chunk(chunk_n * 3);
        std::memcpy(pos_chunk.data(), pos + offset * 3, chunk_n * 3 * sizeof(float));
        result.pos_chunks.push_back(std::move(pos_chunk));

        // Copy x for this chunk: (1, 4, chunk_n)
        // x is channel-major: [ch0_N][ch1_N][ch2_N][ch3_N]
        std::vector<float> x_chunk(4 * chunk_n);
        for (int ch = 0; ch < 4; ++ch) {
            const float* src = x + ch * N + offset;
            float* dst = x_chunk.data() + ch * chunk_n;
            std::memcpy(dst, src, chunk_n * sizeof(float));
        }
        result.x_chunks.push_back(std::move(x_chunk));

        offset += chunk_n;
    }

    return result;
}

// ===========================================================================
// trim_padding
// ===========================================================================
// logits layout in flat memory: (1, 2, padded_N) channel-major:
//   [ch0_0 … ch0_{padded_N-1}, ch1_0 … ch1_{padded_N-1}]
//
// After trim we want: (1, 2, N_true):
//   [ch0_0 … ch0_{N_true-1}, ch1_0 … ch1_{N_true-1}]
//
// Channel 0 is already at the correct offset.  Only channel 1 needs to be
// moved from position padded_N to position N_true.
// ===========================================================================
void SubcloudUtils::trim_padding(
    float* logits,
    int N_true,
    int padded_N)
{
    if (N_true < 0 || padded_N < 0) {
        throw std::invalid_argument(
            "trim_padding: N_true and padded_N must be >= 0");
    }
    if (N_true > padded_N) {
        throw std::invalid_argument(
            "trim_padding: N_true must be <= padded_N");
    }
    if (N_true == padded_N) {
        return;  // nothing to do
    }

    // Move channel 1's first N_true values from offset padded_N → offset N_true
    // Source:      logits + padded_N
    // Destination: logits + N_true
    // Size:        N_true floats
    std::memmove(
        logits + N_true,
        logits + padded_N,
        N_true * sizeof(float));
}
