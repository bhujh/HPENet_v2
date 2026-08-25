# C++ 推理源码修复

## TL;DR

> **Quick Summary**: 修复 C++ TensorRT 推理管线 4 个 bug + 文档化 1 个行为差异。6 步顺序执行，预计 15 分钟。
>
> **Parallel Execution**: NO — 逐个执行

---

## Context

### 发现的问题

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `main.cpp` | 显示文件名错位（缺 shuffle）| 加 shuffle |
| 2 | `main.cpp` | max_n 默认值不一致 | 6000→30000 |
| 3 | `trt_inference.*` + `pipeline.cpp` | FP16 引擎输出 unsafe | 加 is_output_fp16() |
| 4 | `pipeline.cpp` | scatter_mean 双重 memset | 删除冗余 |
| 5 | `pipeline.cpp` | 拆分 vs InstanceNorm 未文档化 | 加注释 |

### 红线
- 不改变核心推理逻辑、不修改 CUDA kernel、不修改 Python、不引入新依赖、不执行 git

---

## 执行顺序（逐个执行）

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6
```

---

## TODOs

- [x] 1. main.cpp — max_n 默认值 6000 → 30000 + 显示逻辑加 shuffle

  **改什么**:
  - 行 35: `int max_n = 6000;` → `int max_n = 30000;`
  - 行 207: `std::sort(...)` 之后加两行:
    ```cpp
    std::mt19937 rng(config.seed);
    std::shuffle(all_ply.begin(), all_ply.end(), rng);
    ```

  **不改**: process_directory、seed 值、其他文件

  **验证**: 读取 main.cpp 确认改动正确

- [x] 2. FP16 安全 — trt_inference.h + trt_inference.cpp + pipeline.cpp

  **改什么**:

  **trt_inference.h**:
  - 加 `nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;` 到 private
  - 加 `bool is_output_fp16() const { return output_dtype_ == nvinfer1::DataType::kHALF; }` 到 public
  - `float* infer(...)` → `void* infer(...)`
  - `float* get_output()` → `void* get_output()`

  **trt_inference.cpp**:
  - 构造函数中保存 `output_dtype_ = engine_->get()->getTensorDataType(output_name_.c_str());`
  - `return static_cast<float*>(d_output_->data());` → `return d_output_->data();`

  **pipeline.cpp** (h_logits 拷贝段):
  - 加 FP16/FP32 分支判断，FP16 时用 `uint16_t` + `__half2float()` 转换

  **不改**: TRT 引擎构建、CUDA kernel、Python 代码

  **验证**: 读取三个文件确认改动

- [x] 3. pipeline.cpp — 删冗余 memset + 加 InstanceNorm 注释

  **改什么**:
  - 删 `d_out.memset(0, 0);` 和 `d_cnt.memset(0, 0);`（scatter_mean.cu 内部已置零）
  - `split_oversized` 调用前加注释:
    ```
    // NOTE: Splitting oversized sub-clouds may produce slightly different results
    // vs Python due to InstanceNorm statistics computed per-chunk.
    ```

  **不改**: launch_scatter_mean_kernel、split_oversized 逻辑

  **验证**: 读取 pipeline.cpp 确认

- [x] 4. 编译验证

  ```bash
  cd deploy/CPP/build && cmake .. && make -j$(nproc)
  ```
  预期: 0 errors, 0 warnings

- [x] 5. GoogleTest 回归

  ```bash
  cd deploy/CPP/build && ctest --output-on-failure
  ```
  预期: 全部通过

- [x] 6. 端到端推理验证

  ```bash
  cd deploy/CPP && ./build/hpenet_trt_infer --num_files=3
  ```
  预期: 输出精度汇总, 无崩溃

---

## 成功标准
- [ ] 编译通过，0 errors/warnings
- [ ] ctest 全部通过
- [ ] 端到端推理运行正常，精度合理
- [ ] 显示文件名与 process_directory 处理顺序一致
- [ ] max_n 三处默认值均为 30000
