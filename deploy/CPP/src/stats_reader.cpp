#include "stats_reader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

// -----------------------------------------------------------------------
// Internal helpers — hand-written JSON parser for a known schema.
// Only supports: {"key": <scalar>, "key": [f1,f2,...], ...}
// -----------------------------------------------------------------------

namespace {

/// Read entire file into a string.
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("StatsReader: cannot open file: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// Trim leading/trailing whitespace in place.
static void trim(std::string& s) {
    // Leading whitespace
    auto first = std::find_if(s.begin(), s.end(),
                               [](unsigned char c) { return !std::isspace(c); });
    // Trailing whitespace
    auto last = std::find_if(s.rbegin(), s.rend(),
                              [](unsigned char c) { return !std::isspace(c); });
    s.erase(last.base(), s.end());
    s.erase(s.begin(), first);
}

/**
 * @brief Find the JSON value string for a given key.
 *
 * Searches for `"key":` in the JSON text and returns the raw value substring,
 * which is either a scalar token or an array "[...]".
 *
 * @param json  The full JSON text.
 * @param key   Key to search for (without quotes).
 * @return Raw value substring (e.g. "0.5" or "[0.1,0.2,0.3]").
 */
static std::string find_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        throw std::runtime_error("StatsReader: key not found: \"" + key + "\"");
    }
    pos += search.size();

    // Skip whitespace after colon
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    if (pos >= json.size()) {
        throw std::runtime_error("StatsReader: unexpected end of JSON after key \"" +
                                 key + "\"");
    }

    if (json[pos] == '[') {
        // Array — find matching ']'
        size_t end = json.find(']', pos);
        if (end == std::string::npos) {
            throw std::runtime_error("StatsReader: unmatched '[' for key \"" +
                                     key + "\"");
        }
        return json.substr(pos, end - pos + 1);
    } else {
        // Scalar — read until ',' or '}' or whitespace
        size_t end = pos;
        while (end < json.size() && json[end] != ',' && json[end] != '}' &&
               !std::isspace(static_cast<unsigned char>(json[end]))) {
            ++end;
        }
        if (end == pos) {
            throw std::runtime_error("StatsReader: empty value for key \"" +
                                     key + "\"");
        }
        return json.substr(pos, end - pos);
    }
}

/**
 * @brief Parse a float array string like "[0.1, 0.2, 0.3]".
 *
 * @param arr_str  The raw array substring including brackets.
 * @param out      Output buffer (must have space for @p expected elements).
 * @param expected Number of elements required.
 */
static void parse_float_array(const std::string& arr_str, float* out,
                              int expected) {
    size_t start = arr_str.find('[');
    size_t end   = arr_str.find(']');
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        throw std::runtime_error("StatsReader: invalid array format");
    }

    std::string content = arr_str.substr(start + 1, end - start - 1);
    std::stringstream ss(content);
    std::string token;
    int count = 0;

    while (std::getline(ss, token, ',')) {
        trim(token);
        if (token.empty()) continue;
        if (count >= expected) {
            throw std::runtime_error(
                "StatsReader: array has more than " + std::to_string(expected) +
                " elements");
        }
        out[count++] = std::stof(token);
    }

    if (count != expected) {
        throw std::runtime_error("StatsReader: array has " + std::to_string(count) +
                                 " elements, expected " + std::to_string(expected));
    }
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// StatsReader public API
// -----------------------------------------------------------------------

FeatureStats StatsReader::load(const std::string& path) {
    const std::string json = read_file(path);
    FeatureStats stats;

    // Parse feat_mean[3]
    const std::string feat_mean_str = find_value(json, "feat_mean");
    parse_float_array(feat_mean_str, stats.feat_mean, 3);

    // Parse feat_std[3]
    const std::string feat_std_str = find_value(json, "feat_std");
    parse_float_array(feat_std_str, stats.feat_std, 3);

    // Parse z_mean scalar
    const std::string z_mean_str = find_value(json, "z_mean");
    stats.z_mean = std::stof(z_mean_str);

    // Parse z_std scalar
    const std::string z_std_str = find_value(json, "z_std");
    stats.z_std = std::stof(z_std_str);

    return stats;
}
