#include "benchmark.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>

// ══════════════════════════════════════════════════════════════════════════════
// GPUTimer
// ══════════════════════════════════════════════════════════════════════════════

GPUTimer::GPUTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

GPUTimer::~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void GPUTimer::start(cudaStream_t stream) {
    cudaEventRecord(start_, stream);
}

void GPUTimer::stop(cudaStream_t stream) {
    cudaEventRecord(stop_, stream);
}

double GPUTimer::elapsed_ms() const {
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return static_cast<double>(ms);
}

// ══════════════════════════════════════════════════════════════════════════════
// CPUTimer
// ══════════════════════════════════════════════════════════════════════════════

void CPUTimer::start() {
    start_ = std::chrono::high_resolution_clock::now();
}

void CPUTimer::stop() {
    stop_ = std::chrono::high_resolution_clock::now();
}

double CPUTimer::elapsed_ms() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
    return static_cast<double>(duration.count()) / 1000.0;
}

// ══════════════════════════════════════════════════════════════════════════════
// Benchmark
// ══════════════════════════════════════════════════════════════════════════════

void Benchmark::record_total(double ms) {
    total_times_.push_back(ms);
}

void Benchmark::record_infer(double ms) {
    infer_times_.push_back(ms);
}

void Benchmark::record_preprocess(double ms) {
    preprocess_times_.push_back(ms);
}

void Benchmark::record_postprocess(double ms) {
    postprocess_times_.push_back(ms);
}

// ── 内部统计辅助 ────────────────────────────────────────────────────────────────

namespace {

/// 计算一组延时数据的统计量
struct LatencyStats {
    double min  = 0.0;
    double max  = 0.0;
    double mean = 0.0;
    double p50  = 0.0;
    double p95  = 0.0;
    double p99  = 0.0;
    bool   valid = false;
};

LatencyStats compute_stats(const std::vector<double>& data) {
    LatencyStats s;
    if (data.empty()) {
        return s;
    }

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());
    const int n = static_cast<int>(sorted.size());

    s.min   = sorted.front();
    s.max   = sorted.back();
    s.mean  = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
    s.p50   = sorted[n / 2];
    s.p95   = sorted[static_cast<int>(n * 0.95)];
    s.p99   = sorted[static_cast<int>(n * 0.99)];
    s.valid = true;

    return s;
}

} // anonymous namespace

// ── compute ─────────────────────────────────────────────────────────────────────

BenchmarkResult Benchmark::compute() const {
    BenchmarkResult r;

    const auto total_stat = compute_stats(total_times_);
    const auto infer_stat = compute_stats(infer_times_);
    const auto pre_stat   = compute_stats(preprocess_times_);
    const auto post_stat  = compute_stats(postprocess_times_);

    // Total latency (full stats)
    r.total_latency_min  = total_stat.min;
    r.total_latency_max  = total_stat.max;
    r.total_latency_mean = total_stat.mean;
    r.total_latency_p50  = total_stat.p50;
    r.total_latency_p95  = total_stat.p95;
    r.total_latency_p99  = total_stat.p99;

    // Infer latency
    r.infer_latency_mean = infer_stat.mean;
    r.infer_latency_min  = infer_stat.min;
    r.infer_latency_max  = infer_stat.max;
    r.infer_latency_p50  = infer_stat.p50;
    r.infer_latency_p95  = infer_stat.p95;
    r.infer_latency_p99  = infer_stat.p99;

    // Preprocess latency
    r.preprocess_latency_mean = pre_stat.mean;
    r.preprocess_latency_min  = pre_stat.min;
    r.preprocess_latency_max  = pre_stat.max;
    r.preprocess_latency_p50  = pre_stat.p50;
    r.preprocess_latency_p95  = pre_stat.p95;
    r.preprocess_latency_p99  = pre_stat.p99;

    // Postprocess latency
    r.postprocess_latency_mean = post_stat.mean;
    r.postprocess_latency_min  = post_stat.min;
    r.postprocess_latency_max  = post_stat.max;
    r.postprocess_latency_p50  = post_stat.p50;
    r.postprocess_latency_p95  = post_stat.p95;
    r.postprocess_latency_p99  = post_stat.p99;

    // Total points and files
    r.total_points = total_points_;
    r.num_files    = num_files_;

    // Throughput: files/sec = num_files / (total_mean_s)
    const double total_mean_s = r.total_latency_mean / 1000.0;
    if (total_mean_s > 0.0) {
        r.throughput_files_per_sec  = num_files_ / total_mean_s;
        r.throughput_points_per_sec = total_points_ / total_mean_s;
    }

    return r;
}

// ── print ───────────────────────────────────────────────────────────────────────

void Benchmark::print(const BenchmarkResult& r) {
    // Helper: 打印一行统计数据
    auto print_row = [&](const char* label,
                         double total_val,
                         double infer_val,
                         double pre_val,
                         double post_val) {
        std::cout << "  " << std::left << std::setw(16) << label
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(9) << total_val
                  << std::setw(9) << infer_val
                  << std::setw(9) << pre_val
                  << std::setw(9) << post_val
                  << "\n";
    };

    std::cout << "\n";
    std::cout << "========== Benchmark Results ==========\n";

    auto fmt_int = [](const char* label, int val) {
        std::cout << "  " << std::left << std::setw(16) << label
                  << std::right << std::setw(12) << val << "\n";
    };
    fmt_int("Files processed:", r.num_files);
    fmt_int("Total points:", r.total_points);

    std::cout << "\n"
              << "Latency (ms):\n"
              << "  " << std::left << std::setw(16) << ""
              << std::right
              << std::setw(9) << "Total"
              << std::setw(9) << "Infer"
              << std::setw(9) << "Pre"
              << std::setw(9) << "Post"
              << "\n";

    print_row("Mean:",  r.total_latency_mean,
              r.infer_latency_mean,
              r.preprocess_latency_mean,
              r.postprocess_latency_mean);

    print_row("Min:",   r.total_latency_min,
              r.infer_latency_min,
              r.preprocess_latency_min,
              r.postprocess_latency_min);

    print_row("Max:",   r.total_latency_max,
              r.infer_latency_max,
              r.preprocess_latency_max,
              r.postprocess_latency_max);

    print_row("P50:",   r.total_latency_p50,
              r.infer_latency_p50,
              r.preprocess_latency_p50,
              r.postprocess_latency_p50);

    print_row("P95:",   r.total_latency_p95,
              r.infer_latency_p95,
              r.preprocess_latency_p95,
              r.postprocess_latency_p95);

    print_row("P99:",   r.total_latency_p99,
              r.infer_latency_p99,
              r.preprocess_latency_p99,
              r.postprocess_latency_p99);

    std::cout << "\n"
              << "  Throughput: "
              << std::fixed << std::setprecision(2)
              << r.throughput_files_per_sec << " files/sec | "
              << std::fixed << std::setprecision(0)
              << r.throughput_points_per_sec << " points/sec\n";

    std::cout << "========================================\n";
    std::cout << std::flush;
}
