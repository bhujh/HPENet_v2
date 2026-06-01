#include "trt_inference.h"

#include <iostream>
#include <stdexcept>
#include <string>

// ============================================================================
//  TrInference 实现
// ============================================================================

TrInference::TrInference(TrEngine& engine, TrLogger& logger)
    : engine_(&engine) {
    (void)logger;  // logger 保留在 API 中以保持接口一致性

    // 1. 创建执行上下文
    context_ = engine_->create_context();
    CHECK_TRT(context_, "Failed to create execution context");

    // 2. 自动发现输出张量名称
    auto outputs = engine_->get_output_tensors();
    if (outputs.empty()) {
        throw std::runtime_error("[TrInference] Engine has no output tensors");
    }
    output_name_ = outputs[0].name;

    // 3. 打印基本信息
    std::cout << "[TrInference] Output tensor: \"" << output_name_ << "\""
              << " (nbDims=" << outputs[0].dims.nbDims << ")"
              << std::endl;
}

TrInference::~TrInference() {
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
}

// ── I/O 形状与地址绑定 ──

void TrInference::set_input_shape(const std::string& name, nvinfer1::Dims dims) {
    bool ok = context_->setInputShape(name.c_str(), dims);
    if (!ok) {
        throw std::runtime_error("[TrInference] setInputShape failed for \""
                                 + name + "\"");
    }
}

void TrInference::set_tensor_address(const std::string& name, void* ptr) {
    if (!ptr) {
        throw std::runtime_error("[TrInference] Null pointer for tensor \""
                                 + name + "\"");
    }
    bool ok = context_->setTensorAddress(name.c_str(), ptr);
    if (!ok) {
        throw std::runtime_error("[TrInference] setTensorAddress failed for \""
                                 + name + "\"");
    }
}

// ── 推理控制 ──

void TrInference::run_async(cudaStream_t stream) {
    bool ok = context_->enqueueV3(stream);
    if (!ok) {
        throw std::runtime_error("[TrInference] enqueueV3 failed");
    }
}

void TrInference::synchronize(cudaStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

// ── 便捷推理 ──

float* TrInference::infer(float* d_pos, float* d_x, int N, cudaStream_t stream) {
    // ---------- 1. 设置动态输入形状 ----------
    //
    // pos: (1, N, 3) — 动态维度在 axis=1
    nvinfer1::Dims pos_shape;
    pos_shape.nbDims = 3;
    pos_shape.d[0] = 1;
    pos_shape.d[1] = N;
    pos_shape.d[2] = 3;

    // x:   (1, 4, N) — 动态维度在 axis=2
    nvinfer1::Dims x_shape;
    x_shape.nbDims = 3;
    x_shape.d[0] = 1;
    x_shape.d[1] = 4;
    x_shape.d[2] = N;

    set_input_shape("pos", pos_shape);
    set_input_shape("x", x_shape);

    // ---------- 2. 绑定输入地址 ----------
    set_tensor_address("pos", d_pos);
    set_tensor_address("x", d_x);

    // ---------- 3. 分配/重用输出 buffer ----------
    // 输出形状: (1, 2, N) float32
    size_t out_size = static_cast<size_t>(1) * 2 * N * sizeof(float);
    if (!d_output_ || d_output_->size() < out_size) {
        // 按需重新分配（比 N 增大时扩容，缩小时重用避免重复分配）
        d_output_ = std::make_unique<CudaBuffer>(out_size);
    }
    set_tensor_address(output_name_, d_output_->data());

    // ---------- 4. 异步推理 ----------
    run_async(stream);

    return static_cast<float*>(d_output_->data());
}
