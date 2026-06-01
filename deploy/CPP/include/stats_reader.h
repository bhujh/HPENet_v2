#pragma once
#include "types.h"
#include <string>

/**
 * @brief Reads feature normalization statistics from a JSON file.
 *
 * The expected JSON format (produced by convert_stats.py):
 * {
 *   "feat_mean": [f1, f2, f3],
 *   "feat_std":  [s1, s2, s3],
 *   "z_mean":    zm,
 *   "z_std":     zs
 * }
 *
 * Uses a hand-written parser (no nlohmann/json dependency).
 */
class StatsReader {
public:
    /**
     * @brief Load FeatureStats from a JSON file.
     * @param path  Path to the JSON stats file.
     * @return Populated FeatureStats struct.
     * @throws std::runtime_error on I/O or parse errors.
     */
    static FeatureStats load(const std::string& path);
};
