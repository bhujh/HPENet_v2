#pragma once
//#include <string>
//#include <memory>
//#include <iostream>
//#include <stdexcept>
//#include <NvInfer.h>
//#include "logger.h"
#include "trt_engine.h"
#include "cuda_utils.h"

/// TrInference — TensorRT 推理执行器
///
/// 功能等价于 Python trt_utils.py:TRTSession.run()。
/// 封装 IExecutionContext + 动态形状 + enqueueV3。
/// 使用 TensorRT 8.6 API（setInputShape / setTensorAddress / enqueueV3）。
///
/// 线程安全: 非线程安全（IExecutionContext 本身不保证线程安全），
/// 每个线程应使用独立实例。
class TrInference {
public:
    /// @param engine 已加载的 TrEngine 实例（生命周期必须长于本对象）
    /// @param logger TrLogger 日志实例（仅用于 API 签名一致性）
    TrInference(TrEngine& engine, TrLogger& logger);
    ~TrInference();

    TrInference(const TrInference&) = delete;
    TrInference& operator=(const TrInference&) = delete;

    // ── I/O 形状与地址绑定 ──

    /// 设置输入张量的动态形状
    void set_input_shape(const std::string& name, nvinfer1::Dims dims);

    /// 绑定 I/O 张量的 GPU 地址
    void set_tensor_address(const std::string& name, void* ptr);

    // ── 推理控制 ──

    /// 执行异步推理（enqueueV3）
    void run_async(cudaStream_t stream);

    /// 等待指定流上的推理完成
    void synchronize(cudaStream_t stream);

    /// 便捷方法: 一步完成端到端推理
    ///
    /// 内部依次调用:
    ///   1. set_input_shape("pos", (1, N, 3)) + set_input_shape("x", (1, 4, N))
    ///   2. set_tensorAddress("pos", d_pos) + set_tensorAddress("x", d_x)
    ///   3. 分配/重用输出 buffer → setTensorAddress("output", ...)
    ///   4. enqueueV3(stream)
    ///
    /// @param d_pos  位置输入, shape (1, N, 3), float32, GPU
    /// @param d_x    特征输入, shape (1, 4, N), float32, GPU
    /// @param N      点数
    /// @param stream CUDA stream (默认 0 = default stream)
    /// @return GPU 上的输出指针, shape (1, 2, N), float32
    void* infer(float* d_pos, float* d_x, int N, cudaStream_t stream = 0);

    // ── 查询 ──

    /// 获取输出 GPU 指针（最近一次 infer 或 set_tensor_address 绑定的地址）
    void* get_output() const {
        return d_output_ ? d_output_->data() : nullptr;
    }

    /// 获取底层 IExecutionContext（高级用法，谨慎修改状态）
    nvinfer1::IExecutionContext* get_context() const { return context_; }

    /// 获取输出张量名称
    const std::string& get_output_name() const { return output_name_; }

    bool is_output_fp16() const { return output_dtype_ == nvinfer1::DataType::kHALF; }

private:
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::unique_ptr<CudaBuffer> d_output_;
    TrEngine* engine_ = nullptr;

    /// 输出张量名称（构造时从 engine 自动发现）
    std::string output_name_;

    nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;
};
