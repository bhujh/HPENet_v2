#pragma once
#include"deployment_ai.h"
#include <iostream>
#include <ctime>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
// #include <fstream>
//#include <cstdlib>
#include <atomic>
//Tensorrt
#include <NvInfer.h>



class DEPLOYAI_LIB_API TRTLogger : public nvinfer1::ILogger {
private:
    std::atomic<Severity> mReportableSeverity;
    std::string mLogFilePath;           // [FIX] 补全遗漏的成员变量
    std::queue<std::string> mLogQueue;
    std::mutex mQueueMutex;
    std::condition_variable mCv;
    std::thread mWorkerThread;
    bool mExit;

    // 辅助函数：将 Severity 枚举转换为字符串
    const char* severityToString(Severity severity) noexcept;

    // 后台工作线程：异步处理日志写入
    void logWorker();

public:
    explicit TRTLogger(
        Severity severity = Severity::kWARNING,
        const std::string& logFilePath = "trt_engine.log");

    ~TRTLogger() override;

    // 核心 log 接口 (noexcept 保证)
    void log(Severity severity, const char* msg) noexcept override;

    // 动态调整日志级别
    void setReportableSeverity(Severity severity) noexcept {
        mReportableSeverity.store(severity, std::memory_order_relaxed);
    }
};


// ── TrLogger ──
// 继承 nvinfer1::ILogger，将 TRT 内部日志输出到 stderr
class DEPLOYAI_LIB_API TrLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};


// ── CHECK_CUDA ──
// 检查 CUDA 调用，失败时打印位置并 abort
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)          \
                      << " | " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

// ── CHECK_LAST_CUDA ──
// 检查上次 CUDA kernel 启动错误 (cudaGetLastError) + 同步
#define CHECK_LAST_CUDA()                                                    \
    do {                                                                     \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[CUDA KERNEL ERROR] " << cudaGetErrorString(err)   \
                      << " | " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::abort();                                                    \
        }                                                                    \
        err = cudaDeviceSynchronize();                                       \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[CUDA SYNC ERROR] " << cudaGetErrorString(err)     \
                      << " | " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

// ── CHECK_TRT ──
// 检查 TRT 返回值（指针是否为空等）
#define CHECK_TRT(call, msg)                                                 \
    do {                                                                     \
        if (!(call)) {                                                       \
            std::cerr << "[TRT ERROR] " << (msg)                             \
                      << " | " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::abort();                                                    \
        }                                                                    \
    } while (0)
