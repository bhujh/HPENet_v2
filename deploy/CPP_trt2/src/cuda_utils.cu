#include "logger.h"
#include "cuda_utils.h"
//#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// CudaBuffer implementation
// ----------------------------------------------------------------------------
CudaBuffer::CudaBuffer(size_t size)
    : data_(nullptr), size_(size) {
    if (size_ > 0) {
        CHECK_CUDA(cudaMalloc(&data_, size_));
    }
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        if (valid()) {
            CHECK_CUDA(cudaFree(data_));
        }
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

CudaBuffer::~CudaBuffer() {
    if (valid()) {
        CHECK_CUDA(cudaFree(data_));
    }
}

void CudaBuffer::upload(const void* host_ptr, size_t size, cudaStream_t stream) {
    CHECK_CUDA(cudaMemcpyAsync(data_, host_ptr, size,
        cudaMemcpyHostToDevice, stream));
}

void CudaBuffer::download(void* host_ptr, size_t size, cudaStream_t stream) {
    CHECK_CUDA(cudaMemcpyAsync(host_ptr, data_, size,
        cudaMemcpyDeviceToHost, stream));
}

void CudaBuffer::memset(int value, size_t offset) {
    CHECK_CUDA(cudaMemset(static_cast<char*>(data_) + offset, value,
        size_ - offset));
}


// ----------------------------------------------------------------------------
// CudaStream implementation
// ----------------------------------------------------------------------------
CudaStream::CudaStream() {
    CHECK_CUDA(cudaStreamCreate(&stream_));
}

CudaStream::~CudaStream() {
    if (stream_) {
        CHECK_CUDA(cudaStreamDestroy(stream_));
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) {
            CHECK_CUDA(cudaStreamDestroy(stream_));
        }
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() {
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

// ----------------------------------------------------------------------------
// DeviceGuard implementation
// ----------------------------------------------------------------------------
DeviceGuard::DeviceGuard(int device)
    : saved_device_(-1) {
    CHECK_CUDA(cudaGetDevice(&saved_device_));
    if (saved_device_ != device) {
        CHECK_CUDA(cudaSetDevice(device));
    }
}

DeviceGuard::~DeviceGuard() {
    if (saved_device_ >= 0) {
        // best‑effort restore in destructor
        cudaSetDevice(saved_device_);
    }
}

// ----------------------------------------------------------------------------
// get_gpu_memory_info
// ----------------------------------------------------------------------------
DEPLOYAI_LIB_API GpuMemoryInfo get_gpu_memory_info(int device) {
    GpuMemoryInfo info;
    DeviceGuard guard(device);
    CHECK_CUDA(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
    info.used_bytes = info.total_bytes - info.free_bytes;
    return info;
}