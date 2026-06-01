#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ── BenchmarkResult ────────────────────────────────────────────────────────────

/// 基准测试统计结果
struct BenchmarkResult {
    // 延迟统计 (ms)
    double total_latency_min   = 0.0;
    double total_latency_max   = 0.0;
    double total_latency_mean  = 0.0;
    double total_latency_p50   = 0.0;
    double total_latency_p95   = 0.0;
    double total_latency_p99   = 0.0;

    // 各阶段延迟均值 (ms)
    double infer_latency_mean       = 0.0;
    double preprocess_latency_mean  = 0.0;
    double postprocess_latency_mean = 0.0;

    // 各阶段延迟统计 (ms) — 仅 total 有完整统计
    double infer_latency_min       = 0.0;
    double infer_latency_max       = 0.0;
    double infer_latency_p50       = 0.0;
    double infer_latency_p95       = 0.0;
    double infer_latency_p99       = 0.0;

    double preprocess_latency_min  = 0.0;
    double preprocess_latency_max  = 0.0;
    double preprocess_latency_p50  = 0.0;
    double preprocess_latency_p95  = 0.0;
    double preprocess_latency_p99  = 0.0;

    double postprocess_latency_min = 0.0;
    double postprocess_latency_max = 0.0;
    double postprocess_latency_p50 = 0.0;
    double postprocess_latency_p95 = 0.0;
    double postprocess_latency_p99 = 0.0;

    // 吞吐量
    double throughput_files_per_sec  = 0.0;
    double throughput_points_per_sec = 0.0;

    // 总处理点数
    int total_points = 0;
    int num_files    = 0;
};

// ── GPUTimer ───────────────────────────────────────────────────────────────────

/// 基于 CUDA event 的 GPU 精确计时器
///
/// 使用 cudaEventCreate / cudaEventRecord / cudaEventElapsedTime 测量
/// GPU 端操作的真实耗时（不同于 std::chrono 的 CPU 端测量）。
class GPUTimer {
public:
    GPUTimer();
    ~GPUTimer();

    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    /// 记录起始 event
    void start(cudaStream_t stream = 0);

    /// 记录终止 event
    void stop(cudaStream_t stream = 0);

    /// 返回最后一次 start->stop 的毫秒数
    /// 内部调用 cudaEventSynchronize 确保同步
    double elapsed_ms() const;

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// ── CPUTimer ───────────────────────────────────────────────────────────────────

/// 基于 std::chrono 的 CPU 计时器
class CPUTimer {
public:
    void start();
    void stop();
    double elapsed_ms() const;

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point stop_;
};

// ── Benchmark ──────────────────────────────────────────────────────────────────

/// 性能基准模块：采集多次推理的各阶段延迟并输出统计数据
///
/// 用法:
///   Benchmark bench;
///   for (...) {
///       GPUTimer gpu_timer;
///       gpu_timer.start();
///       // ... 推理 ...
///       gpu_timer.stop();
///       bench.record_total(gpu_timer.elapsed_ms());
///   }
///   bench.set_num_files(N);
///   BenchmarkResult r = bench.compute();
///   Benchmark::print(r);
class Benchmark {
public:
    /// 记录一次完整的端到端延迟 (ms)
    void record_total(double ms);

    /// 记录一次推理延迟 (ms) — 仅 enqueueV3 时间
    void record_infer(double ms);

    /// 记录一次预处理延迟 (ms)
    void record_preprocess(double ms);

    /// 记录一次后处理延迟 (ms) — scatter_mean + argmax
    void record_postprocess(double ms);

    void set_total_points(int n) { total_points_ = n; }
    void set_num_files(int n) { num_files_ = n; }

    /// 基于所有采样数据计算统计结果
    BenchmarkResult compute() const;

    /// 打印格式化的基准测试结果
    static void print(const BenchmarkResult& result);

private:
    std::vector<double> total_times_;
    std::vector<double> infer_times_;
    std::vector<double> preprocess_times_;
    std::vector<double> postprocess_times_;
    int total_points_ = 0;
    int num_files_    = 0;
};
