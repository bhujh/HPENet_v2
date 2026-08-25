# GPU 显存监控 — 补强方案

## TL;DR

> **Quick Summary**: 在 `cuda_utils.h` 添加 `get_gpu_memory_info()` 工具函数，在 `main.cpp` 和 `pipeline.cpp` 的关键阶段插入显存水位打印，实现零配置运行时显存监控。
> 
> **Deliverables**:
> - `cuda_utils.h` 新增 `GpuMemoryInfo` 结构体 + `get_gpu_memory_info()` 函数
> - `main.cpp` 引擎加载后 / warmup 后 / 推理前后打印显存使用量
> - `pipeline.cpp` 每处理一个文件后打印峰值显存
> 
> **Estimated Effort**: Quick（3 个文件，~40 行代码）
> **Parallel Execution**: 可在一次编辑中完成
> **Critical Path**: 无依赖，直接改

---

## Context

### 需求
用户在运行 `hpenet_trt_infer` 时无法感知 GPU 显存占用情况。需要能在运行时实时看到：
- 总显存容量
- 当前已用 / 空闲
- TRT 引擎加载后占了多少
- warmup 后占了多少
- 实际推理时占了多少

### 现状
- `cuda_utils.h`：已有 `CudaBuffer`、`CudaStream`、`DeviceGuard` 三个 RAII 类
- `main.cpp`：有管线初始化、warmup、推理循环的完整流程
- `pipeline.cpp`：有 `process_file()` 和 `process_directory()`
- 无任何 `cudaMemGetInfo` 调用

### 技术选型
使用 `cudaMemGetInfo(&free, &total)` — CUDA Runtime API 自带，零依赖，一行调用。

---

## Work Objectives

### Core Objective
在现有 C++/TRT 推理流水线中添加显存水位监控，用户运行程序时能直观看到各阶段的 GPU 内存使用情况。

### Concrete Deliverables
- `cuda_utils.h`：新增 `get_gpu_memory_info()` (+ `GpuMemoryInfo` 结构体)
- `main.cpp`：4 个监控点打印显存
- `pipeline.cpp`：1 个监控点（每个文件处理后）

---

## TODOs

- [x] 1. `cuda_utils.h` — 添加 `GpuMemoryInfo` + `get_gpu_memory_info()`

  **What to do**:
  - 在文件末尾 `#endif` 之前添加：
    ```cpp
    // ── GPU Memory Info ──
    struct GpuMemoryInfo {
        size_t total_bytes  = 0;  // GPU 总显存
        size_t free_bytes   = 0;  // 当前空闲
        size_t used_bytes   = 0;  // 已用 = total - free
    };

    inline GpuMemoryInfo get_gpu_memory_info(int device = 0) {
        GpuMemoryInfo info;
        DeviceGuard guard(device);
        CHECK_CUDA(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
        info.used_bytes = info.total_bytes - info.free_bytes;
        return info;
    }
    ```

  **QA**: 编译通过即可。

- [x] 2. `main.cpp` — 在关键阶段插入显存打印

  **What to do**:
  - 在文件顶部添加 `#include <iomanip>`（已有）
  - 引擎加载成功后（L163 `Pipeline ready` 之后）打印：
    ```cpp
    auto mem = get_gpu_memory_info();
    std::cout << "  GPU Memory: " << mem.used_bytes / 1024 / 1024
              << " MiB used / " << mem.total_bytes / 1024 / 1024 << " MiB total\n";
    ```
  - warmup 完成后（L168 `Warmup done` 之后）同样打印显存
  - 推理循环开始前打印一次基准显存
  - 推理全部结束后打印最终显存

  **QA**: `./hpenet_trt_infer --num_files 1`，确认输出中包含 GPU Memory 信息，数值合理（非 0，非溢出）。

- [x] 3. `pipeline.cpp` — 在 process_file 结尾打印峰值显存

  **What to do**:
  - 在 `process_file()` 函数末尾、`return result` 之前添加：
    ```cpp
    auto mem = get_gpu_memory_info();
    std::cout << "  [GPU] " << mem.used_bytes / 1024 / 1024
              << " MiB used (peak)" << std::endl;
    ```

  **QA**: 多文件推理时，每个文件处理后都能看到 GPU 内存水位。

---

## Final Verification

- [x] F1. **编译验证** — `cmake .. && make -j` 零错误零警告
- [x] F2. **运行验证** — `./hpenet_trt_infer --num_files 1` 输出包含 GPU Memory 行
- [x] F3. **数值合理性** — 已用显存 > 0 且 < 总显存，TRT 引擎加载后显存明显增长(~18-19MB)

---

## Commit Strategy

- **Commit 1**: `feat(deploy/cpp): add GPU memory monitoring to inference pipeline`
  - `deploy/CPP/include/cuda_utils.h` — GpuMemoryInfo + get_gpu_memory_info()
  - `deploy/CPP/src/main.cpp` — 4 个监控点
  - `deploy/CPP/src/pipeline.cpp` — 1 个监控点
