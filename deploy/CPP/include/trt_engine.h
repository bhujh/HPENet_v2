#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <NvInfer.h>
#include "logger.h"

/// TensorInfo — 单个 I/O 张量的元信息
struct TensorInfo {
    std::string name;
    bool is_input;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
};

/// TrEngine — TensorRT 引擎加载器
///
/// 功能等价于 Python trt_utils.py:TRTSession.__init__()。
/// 反序列化 .engine 文件并查询所有 I/O 张量的名称、形状、数据类型。
///
/// 使用 TensorRT 8.6 API（getIOTensorName / getTensorIOMode / getTensorShape / getTensorDataType）。
/// 不使用已废弃的 getBindingIndex / enqueue 等旧 API。
class TrEngine {
public:
    /// 从 .engine 文件反序列化构建 TrEngine
    /// @param engine_path 序列化引擎文件路径
    /// @param logger      TrLogger 日志实例
    /// @throws std::runtime_error 文件不存在或反序列化失败
    TrEngine(const std::string& engine_path, TrLogger& logger);

    ~TrEngine();

    // 禁止拷贝
    TrEngine(const TrEngine&) = delete;
    TrEngine& operator=(const TrEngine&) = delete;

    /// 获取原始 ICudaEngine 指针
    nvinfer1::ICudaEngine* get() const { return engine_; }

    /// 获取所有 I/O 张量信息列表
    const std::vector<TensorInfo>& get_tensors() const { return tensors_; }

    /// 获取所有输入张量信息
    std::vector<TensorInfo> get_input_tensors() const;

    /// 获取所有输出张量信息
    std::vector<TensorInfo> get_output_tensors() const;

    /// 创建执行上下文
    nvinfer1::IExecutionContext* create_context() const;

    /// 打印所有 I/O 张量的名称、形状、数据类型
    void print_info() const;

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    std::vector<TensorInfo> tensors_;
};
