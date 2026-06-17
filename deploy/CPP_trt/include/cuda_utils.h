// ============================================================================
// HPENet V2 — GPU Memory RAII Manager
//
// Header-only CUDA RAII wrappers:
//   - CudaBuffer  : cudaMalloc / cudaFree with upload/download/memset
//   - CudaStream  : cudaStreamCreate / cudaStreamDestroy
//   - DeviceGuard : cudaSetDevice save/restore
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include "logger.h"

// ============================================================================
// CudaBuffer — RAII wrapper for GPU memory
//
// Automatically allocates on construction, frees on destruction.
// Supports move semantics; no copying allowed.
// ============================================================================
class CudaBuffer {
public:
    /// Allocate GPU memory of given size (in bytes).
    explicit CudaBuffer(size_t size)
        : data_(nullptr), size_(size) {
        if (size_ > 0) {
            CHECK_CUDA(cudaMalloc(&data_, size_));
        }
    }

    /// Allocate GPU memory for N elements of type T.
    template <typename T>
    explicit CudaBuffer(int N) : CudaBuffer(N * sizeof(T)) {}

    // --- non-copyable ---
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // --- movable ---
    CudaBuffer(CudaBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            // free existing resource
            if (valid()) {
                CHECK_CUDA(cudaFree(data_));
            }
            // transfer ownership
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~CudaBuffer() {
        if (valid()) {
            CHECK_CUDA(cudaFree(data_));
        }
    }

    // --- accessors ---
    void* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }

    // --- host↔device transfers ---

    /// Upload data from host to GPU. (cudaMemcpyAsync, Host→Device)
    void upload(const void* host_ptr, size_t size, cudaStream_t stream = 0) {
        CHECK_CUDA(cudaMemcpyAsync(data_, host_ptr, size,
                                   cudaMemcpyHostToDevice, stream));
    }

    /// Download data from GPU to host. (cudaMemcpyAsync, Device→Host)
    void download(void* host_ptr, size_t size, cudaStream_t stream = 0) {
        CHECK_CUDA(cudaMemcpyAsync(host_ptr, data_, size,
                                   cudaMemcpyDeviceToHost, stream));
    }

    /// Set GPU memory to a constant byte value.
    void memset(int value = 0, size_t offset = 0) {
        CHECK_CUDA(cudaMemset(static_cast<char*>(data_) + offset, value,
                              size_ - offset));
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// CudaStream — RAII wrapper for CUDA stream
//
// Creates stream on construction, destroys on destruction.
// Supports move semantics; no copying allowed.
// ============================================================================
class CudaStream {
public:
    CudaStream() {
        CHECK_CUDA(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_) {
            CHECK_CUDA(cudaStreamDestroy(stream_));
        }
    }

    // --- non-copyable ---
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // --- movable ---
    CudaStream(CudaStream&& other) noexcept
        : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                CHECK_CUDA(cudaStreamDestroy(stream_));
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    void synchronize() {
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    cudaStream_t native() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

// ============================================================================
// DeviceGuard — RAII device switch
//
// Saves the current CUDA device on construction, switches to the target
// device, and restores the original device on destruction.
// ============================================================================
class DeviceGuard {
public:
    explicit DeviceGuard(int device)
        : saved_device_(-1) {
        CHECK_CUDA(cudaGetDevice(&saved_device_));
        if (saved_device_ != device) {
            CHECK_CUDA(cudaSetDevice(device));
        }
    }

    ~DeviceGuard() {
        if (saved_device_ >= 0) {
            // restore — best-effort in destructor
            cudaSetDevice(saved_device_);
        }
    }

    // --- non-copyable (also non-movable) ---
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int saved_device_ = -1;
};

// ── GPU Memory Info ──
struct GpuMemoryInfo {
    size_t total_bytes = 0;
    size_t free_bytes  = 0;
    size_t used_bytes  = 0;  // total - free
};

inline GpuMemoryInfo get_gpu_memory_info(int device = 0) {
    GpuMemoryInfo info;
    DeviceGuard guard(device);
    CHECK_CUDA(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
    info.used_bytes = info.total_bytes - info.free_bytes;
    return info;
}
