#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// scatter_mean: 对 src 按 index 分组并计算均值 (dim=0, reduce='mean')
//
// 等价于 torch_scatter.scatter(src, index, dim=0, reduce='mean')
//
// 两阶段 CUDA kernel 实现:
//   Phase 1 — atomicAdd 累加各组 sum 和 count
//   Phase 2 — 逐元素除以 count 得到均值
//
// src:     (N_src, C) float, 输入数据
// index:   (N_src,) int64_t, 分组索引 (0..num_classes-1, 负值跳过)
// output:  (num_classes, C) float, 输出 (预先置零)
// count:   (num_classes,) int, 每组的计数 (预先置零)
// num_src: int, N_src
// num_classes: int, output 第一维大小 (index 最大值+1)
// C:       int, 特征维度
// stream:  cudaStream_t, 默认 0
void launch_scatter_mean_kernel(
    const float* src,
    const int64_t* index,
    float* output,
    int* count,
    int num_src,
    int num_classes,
    int C,
    cudaStream_t stream = 0
);
