#include "trt_engine.h"

#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>

// ---------------------------------------------------------------------------
//  TrEngine 实现
// ---------------------------------------------------------------------------

TrEngine::TrEngine(const std::string& engine_path, TrLogger& logger) {
    // 1. 读取 .engine 文件
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> serialized(size);
    if (!file.read(reinterpret_cast<char*>(serialized.data()), size)) {
        throw std::runtime_error("Failed to read engine file: " + engine_path);
    }
    file.close();

    // 2. 创建 runtime
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    // 3. 反序列化引擎
    engine_ = runtime_->deserializeCudaEngine(serialized.data(), serialized.size());
    if (!engine_) {
        // runtime_ 会在析构中清理
        throw std::runtime_error("Failed to deserialize engine from: " + engine_path);
    }

    // 4. 遍历 I/O 张量（TensorRT 8.6 API）
    int nb_tensors = engine_->getNbIOTensors();
    tensors_.reserve(nb_tensors);
    for (int i = 0; i < nb_tensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (!name) {
            continue;
        }

        TensorInfo info;
        info.name = name;
        info.is_input = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        info.dims = engine_->getTensorShape(name);
        info.dtype = engine_->getTensorDataType(name);
        tensors_.push_back(info);
    }
}

TrEngine::~TrEngine() {
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}

std::vector<TensorInfo> TrEngine::get_input_tensors() const {
    std::vector<TensorInfo> inputs;
    for (const auto& t : tensors_) {
        if (t.is_input) {
            inputs.push_back(t);
        }
    }
    return inputs;
}

std::vector<TensorInfo> TrEngine::get_output_tensors() const {
    std::vector<TensorInfo> outputs;
    for (const auto& t : tensors_) {
        if (!t.is_input) {
            outputs.push_back(t);
        }
    }
    return outputs;
}

nvinfer1::IExecutionContext* TrEngine::create_context() const {
    if (!engine_) {
        throw std::runtime_error("Engine not loaded, cannot create execution context");
    }
    return engine_->createExecutionContext();
}

// ── 辅助：dtype → 字符串 ──
static const char* dtype_str(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:   return "float32";
    case nvinfer1::DataType::kHALF:    return "float16";
    case nvinfer1::DataType::kINT8:    return "int8";
    case nvinfer1::DataType::kINT32:   return "int32";
    case nvinfer1::DataType::kBOOL:    return "bool";
    default:                           return "unknown";
    }
}

// ── 辅助：Dims → 字符串 ──
static std::string dims_str(const nvinfer1::Dims& dims) {
    std::string s = "(";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) s += ",";
        s += std::to_string(dims.d[i]);
    }
    s += ")";
    return s;
}

void TrEngine::print_info() const {
    std::cout << "=== TrEngine I/O Tensors ===" << std::endl;
    for (const auto& t : tensors_) {
        std::cout << "  " << (t.is_input ? "INPUT " : "OUTPUT")
                  << "  " << t.name
                  << "  shape=" << dims_str(t.dims)
                  << "  dtype=" << dtype_str(t.dtype)
                  << std::endl;
    }
    std::cout << "============================" << std::endl;
}
