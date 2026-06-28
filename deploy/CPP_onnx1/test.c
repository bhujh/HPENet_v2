#include <stdio.h>
#include <stdlib.h>
// 包含修改后的头文件
#include "onnx_inference_wrapper.h"
//#include "onnx_inference.h"




int main() {
    // 1. 创建推理器
    // 参数: model.onnx, stats.json, min_n, max_n, voxel_size, seed
    OnnxInferencePipeline_C* pipe = onnx_pipeline_create(
        "D:/ProgramData/hpenet_onnx/onnx_model.onnx",
        "D:/ProgramData/hpenet_onnx/stats.json"
    );

    if (!pipe) {
        printf("Error: Failed to create pipeline. Check model paths.\n");
        return -1;
    }

    // 2. 处理文件
    const char* input_file = "D:/ProgramData/hpenet_onnx/radarfull/raw/0000073.ply";
    const char* output_file = "D:/ProgramData/hpenet_onnx/radarfull/output/0000073.ply";

    printf("Processing %s ...\n", input_file);
    float latency = onnx_pipeline_process_file(pipe, input_file, output_file);

    if (latency < 0) {
        // 获取具体的错误信息
        const char* err = onnx_get_last_error(pipe);
        printf("Error: Inference failed. Code: %f, Details: %s\n", latency, err);
    }
    else {
        printf("Success! Inference latency: %.2f ms\n", latency);
        printf("Labeled result saved to: %s\n", output_file);
    }

    // 3. 释放资源
    onnx_pipeline_destroy(pipe);
    printf("Resources cleaned up. Exiting.\n");

    return 0;
}