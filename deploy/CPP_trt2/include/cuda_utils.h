// ============================================================================
// HPENet V2 — GPU Memory RAII Manager
//
// Header-only CUDA RAII wrappers:
//   - CudaBuffer  : cudaMalloc / cudaFree with upload/download/memset
//   - CudaStream  : cudaStreamCreate / cudaStreamDestroy
//   - DeviceGuard : cudaSetDevice save/restore
// ============================================================================
#pragma once
#include"deployment_ai.h"
#include <cuda_runtime.h>
//#include "logger.h"

// ============================================================================
// CudaBuffer — RAII wrapper for GPU memory
//
// Automatically allocates on construction, frees on destruction.
// Supports move semantics; no copying allowed.
// ============================================================================
class DEPLOYAI_LIB_API CudaBuffer {
public:
    /// Allocate GPU memory of given size (in bytes).
    explicit CudaBuffer(size_t size);

    /// Allocate GPU memory for N elements of type T.
    template <typename T>
    explicit CudaBuffer(int N) : CudaBuffer(N * sizeof(T)) {}

    // --- non-copyable ---
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // --- movable ---
    CudaBuffer(CudaBuffer&& other) noexcept;

    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    ~CudaBuffer();

    // --- accessors ---
    void* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }

    // --- host↔device transfers ---

    /// Upload data from host to GPU. (cudaMemcpyAsync, Host→Device)
    void upload(const void* host_ptr, size_t size, cudaStream_t stream = 0);

    /// Download data from GPU to host. (cudaMemcpyAsync, Device→Host)
    void download(void* host_ptr, size_t size, cudaStream_t stream = 0);

    /// Set GPU memory to a constant byte value.
    void memset(int value = 0, size_t offset = 0);

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
class DEPLOYAI_LIB_API CudaStream {
public:
    CudaStream();

    ~CudaStream();

    // --- non-copyable ---
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // --- movable ---
    CudaStream(CudaStream&& other) noexcept;

    CudaStream& operator=(CudaStream&& other) noexcept;

    void synchronize();

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
class DEPLOYAI_LIB_API DeviceGuard {
public:
    explicit DeviceGuard(int device);

    ~DeviceGuard();

    // --- non-copyable (also non-movable) ---
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int saved_device_ = -1;
};

// ── GPU Memory Info ──
struct GpuMemoryInfo {
    size_t total_bytes = 0;
    size_t free_bytes = 0;
    size_t used_bytes = 0;  // total - free
};

DEPLOYAI_LIB_API GpuMemoryInfo get_gpu_memory_info(int device = 0);
