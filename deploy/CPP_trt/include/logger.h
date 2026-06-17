#pragma once
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <NvInfer.h>

// ── TrLogger ──
// 继承 nvinfer1::ILogger，将 TRT 内部日志输出到 stderr
class TrLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            std::cerr << "[TRT ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cerr << "[TRT WARN]  " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[TRT INFO]  " << msg << std::endl;
            break;
        default: // kVERBOSE 等静默
            break;
        }
    }
};

// ── CHECK_CUDA ──
// 检查 CUDA 调用，失败时打印位置并 abort
#define CHECK_CUDA(call)                                                    \
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
