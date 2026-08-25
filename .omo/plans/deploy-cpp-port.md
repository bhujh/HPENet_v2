# HPENet V2 TensorRT 部署 C++ 移植方案

## TL;DR

> **Quick Summary**: 将 deploy/ 中已有 Python TensorRT 部署代码完整移植为 C++/CUDA 独立可执行程序，保存在 deploy/CPP/，实现从 PLY 加载到 TRT 推理到结果输出的全流水线。使用 CMake + C++17 + tinyply + TensorRT 8.6 C++ API。
> 
> **Deliverables**:
> - deploy/CPP/ 完整目录结构（CMakeLists.txt, src/, include/, tests/）
> - 纯 C++ 可执行文件 `hpenet_trt_infer`（加载 .engine 文件运行推理）
> - 配套 Python 辅助脚本（特征统计转换、测试黄金数据生成）
> - GoogleTest 单元测试套件
> - 性能基准输出
> 
> **Estimated Effort**: Large（18 个实现任务 + 4 个最终验证任务）
> **Parallel Execution**: YES - 4 波执行（Wave 1: 7 并行, Wave 2: 5 并行, Wave 3: 2+1, Wave 4: 3 并行）
> **Critical Path**: T1 → T3 → T8 → T9 → T16 → T18

---

## Context

### Original Request
将 deploy/ 中已有的基于 TensorRT 的 Python 部署代码转成 C++/CUDA/TensorRT 实现，保存在 deploy/CPP/ 文件夹。

### Interview Summary
**Key Discussions**:
- **代码形态**：纯 C++ 可执行文件，无 Python 绑定
- **构建系统**：CMake，C++17 标准，目标 Ampere+ GPU (sm80+)
- **推理范围**：仅推理（加载已有 .engine 文件），不包含 ONNX 导出和 TRT 引擎构建
- **PLY 解析**：轻量级 header-only 库 tinyply 3.0，支持混合格式（ASCII + Binary）
- **特征统计**：将 Python .pth 文件转换为 JSON 通用格式供 C++ 读取
- **测试策略**：TDD，以 Python 版本输出作为黄金数据进行逐元素对比验证
- **交付范围**：推理流水线 + 性能基准
- **随机种子**：精确复现 numpy.random.seed(100) 的 shuffle 行为
- **超大子云**：自动分块处理（每块 ≤ max_n=30000）

**Research Findings**:
- **现有 C++ 基础设施**：openpoints/cpp/ 下有 47 个 C++/CUDA 文件，采用 PyTorch Extension 方式构建（setup.py + CUDAExtension），典型的 3 文件模板（.cpp wrapper + .cu kernel + .h header）
- **ONNX 模型已"干净"**：5 个自定义 CUDA 算子（FPS, ball_query, grouping, three_nn, three_interpolate）在 ONNX 导出前已被纯 PyTorch 实现替代；InstanceNorm1d 已被替换为手动归一化；动态 If 节点已移除。C++ 端无需处理自定义算子
- **TensorRT 8.6.1.6 C++ API**：使用 nvinfer1::IRuntime::deserializeCudaEngine(), IExecutionContext::setInputShape(), setTensorAddress(), enqueueV3()
- **体素化算法**：FNV64-1A 哈希 + np.unique + argsort；子云分割使用 count.max() 次迭代取每体素第 i 个点
- **scatter_mean**：C++ 需实现 CUDA kernel 替代 torch_scatter.scatter (dim=0, reduce='mean')

### Metis Review
**Identified Gaps**（已处理）:
- **特征统计格式**：确认为 dict{feat_mean(3,), feat_std(3,), z_mean(scalar), z_std(scalar)} → 转换为 JSON
- **PLY 格式**：确认混合格式 → tinyply 3.0 原生支持
- **GPU 架构精确值**：确认 Ampere+ → CMake 编译参数 sm80, sm86, sm89
- **TensorRT 版本一致性**：确认使用相同 TensorRT 8.6.1.6
- **黄金数据**：需要生成 → 添加 Python 辅助脚本（Wave 1 T6）
- **随机种子复现**：需要匹配 numpy MT19937 → 添加独立实现任务（Wave 1 T7）
- **超大子云处理**：自动分块 → 添加 sub-cloud chunking 任务（Wave 2 T11）
- **scatter_mean 精度风险**：CUDA atomicAdd 浮点累加顺序差异 → 接受 < 1e-4 阈值
- **体素化哈希差异风险**：uint64 溢出语义 → FNV 测试向量验证

---

## Work Objectives

### Core Objective
构建独立的 C++/CUDA/TensorRT 可执行程序，从雷达 PLY 点云文件加载数据，使用预构建的 TensorRT .engine 文件进行 HPENet V2 语义分割推理，输出与 Python 版本等价（逐元素误差 < 1e-4，分类 100% 一致）的预测结果。

### Concrete Deliverables
- `deploy/CPP/` 完整项目（CMakeLists.txt, src/, include/, tests/）
- 可执行文件 `hpenet_trt_infer`
- 功能模块：PLY 读取器、FNV 体素化、特征归一化、scatter_mean、TRT 推理管道
- GoogleTest 单元测试套件（每个模块对应一个测试文件）
- Python 辅助脚本：pth→JSON 转换、黄金数据生成
- 性能基准输出（延迟、吞吐量）

### Definition of Done
- [ ] `mkdir -p deploy/CPP/build && cd deploy/CPP/build && cmake .. && make -j` 构建成功
- [ ] `ctest --output-on-failure` 所有测试通过
- [ ] 端到端测试：给定相同 .engine + .ply + stats.json，C++ 输出 logits 与 Python 输出逐元素最大绝对误差 < 1e-4
- [ ] 端到端测试：C++ argmax 分类结果与 Python 版本 100% 一致
- [ ] `cuda-memcheck ./hpenet_trt_infer ...` 零错误
- [ ] C++ 推理延迟 ≤ Python 版本延迟的 80%
- [ ] 支持 FP32 和 FP16 两种 .engine 文件

### Must Have
- CMake 构建系统，C++17 标准
- CUDA 11.3+ 兼容，目标架构 sm80, sm86, sm89
- TensorRT 8.6.1.6 C++ API（nvinfer1::IRuntime, IExecutionContext, enqueueV3）
- tinyply 3.0 header-only PLY 解析（支持 binary + ASCII）
- FNV64-1A 哈希 CUDA kernel（与 Python 位精确一致）
- 体素化 CUDA kernel（0.1m voxel size, mode=1）
- scatter_mean CUDA kernel（atomicAdd + 计数 + 除法）
- RAII 封装所有 GPU 资源（CudaBuffer, CudaStream）
- 特征统计 JSON 读取器
- numpy.random.seed(100) 兼容的随机数生成器
- FP32 和 FP16 engine 支持
- GoogleTest 单元测试框架
- Python 输出对比验证脚本

### Must NOT Have (Guardrails)
- **不要修改** deploy/ 下任何现有 .py 文件
- **不要引入** OpenCV、PCL、Eigen、Boost 等重型第三方库（CUDA/TensorRT/tinyply 除外）
- **不要实现** ONNX 导出或 TRT 引擎构建功能
- **不要实现** Python 绑定（pybind11）或多 GPU 推理
- **不要修改** openpoints/ 下的模型代码
- **避免过度抽象**：不需要为每类 CUDA kernel 创建工厂模式或虚基类
- **避免注释膨胀**：代码自文档化优先，仅在非直觉处添加必要注释

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - 所有验证均由代理自动执行，无例外。

### Test Decision
- **Infrastructure exists**: NO（仓库无 C++ 单元测试基础设施）
- **Automated tests**: YES — 新建立 GoogleTest 框架 + CTest 集成
- **Framework**: GoogleTest (gtest) + CTest
- **Approach**: TDD — 每个 CUDA/功能模块先编写测试（使用 Python 生成的黄金数据），再实现

### QA Policy
每个任务必须包含代理可执行的 QA 场景（见下方 TODO 模板）。
证据保存至 `.omo/evidence/task-{N}-{scenario-slug}.{ext}`。

- **C++ 编译/测试**：使用 Bash 运行 cmake, make, ctest 命令
- **CUDA kernel 验证**：使用 Bash 运行单元测试，比对黄金数据
- **TRT 推理验证**：使用 Bash 运行 C++ 可执行文件 + Python 对比脚本
- **内存安全**：使用 Bash 运行 cuda-memcheck
- **性能基准**：使用 Bash 运行 benchmark 模式，采集延迟/吞吐量数据

---

## Execution Strategy

### Parallel Execution Waves

> 最大化吞吐：将无依赖任务分组到并行波中。每波完成后才进入下一波。
> 目标：每波 5-7 个任务（Wave 3 因依赖链只有 3 个）。

```
Wave 1 (Start Immediately - 项目基础 + 数据准备, 7 并行):
├── T1: 项目脚手架 + CMakeLists.txt [quick]
├── T2: Logger + 错误检查宏 [quick]
├── T3: 通用类型定义 [quick]
├── T4: PLY 读取器 (tinyply) [unspecified-high]
├── T5: 特征统计 JSON 读取器 + Python 转换脚本 [quick]
├── T6: 黄金数据生成器 (Python) [quick]
└── T7: 随机数工具 (numpy 兼容) [deep]

Wave 2 (After Wave 1 - CUDA Kernel TDD, 5 并行):
├── T8: FNV64-1A 哈希 kernel + 测试 [deep]
├── T9: 体素化 kernel + 测试 [deep]
├── T10: 特征归一化 + 测试 [deep]
├── T11: 子云填充/分块 + 测试 [quick]
└── T12: Scatter-mean kernel + 测试 [deep]

Wave 3 (After Wave 1 - TRT 集成, 2 并行 → 1 串行):
├── T13: GPU 内存 RAII 管理器 [deep] ────┐(并行)
├── T14: TRT 引擎加载器 [unspecified-high] ┘
└── T15: TRT 推理执行器 [deep] ← 依赖 T13, T14

Wave 4 (After Waves 2+3 - 流水线集成, 3 并行):
├── T16: 主推理流水线 [deep]
├── T17: CLI + 主入口 [quick]
└── T18: 性能基准模块 [unspecified-high]

Wave FINAL (After ALL tasks — 4 并行审查):
├── F1: 方案合规审查 (oracle)
├── F2: 代码质量审查 (unspecified-high)
├── F3: 实地手动 QA (unspecified-high)
└── F4: 范围保真度检查 (deep)
→ 呈现结果 → 获得用户明确确认

Critical Path: T1 → T3 → T8 → T9 → T16 → T18
Parallel Speedup: ~65% 比纯串行快
Max Concurrent: 7 (Wave 1), 5 (Wave 2)
```

### Dependency Matrix

| 任务 | 依赖 | 被依赖 | 波 |
|------|------|--------|-----|
| T1 | - | T2-T18, F1-F4 | 1 |
| T2 | T1 | T14, T15 | 1 |
| T3 | T1 | T4, T5, T7, T8, T9, T10, T11, T12, T17 | 1 |
| T4 | T1, T3 | T9, T16 | 1 |
| T5 | T1, T3 | T10, T16 | 1 |
| T6 | - | T8, T9, T12 | 1 |
| T7 | T1, T3 | T9 | 1 |
| T8 | T1, T3, T6 | T9 | 2 |
| T9 | T1, T3, T4, T6, T7, T8 | T16 | 2 |
| T10 | T1, T3, T5 | T16 | 2 |
| T11 | T1, T3 | T16 | 2 |
| T12 | T1, T3, T6 | T16 | 2 |
| T13 | T1 | T15, T16 | 3 |
| T14 | T1, T2 | T15, T16 | 3 |
| T15 | T1, T2, T13, T14 | T16, T18 | 3 |
| T16 | T4, T5, T9, T10, T11, T12, T13, T15 | T17, T18 | 4 |
| T17 | T3, T16 | - | 4 |
| T18 | T15, T16 | - | 4 |

### Agent Dispatch Summary

- **Wave 1**: **7** — T1-T3 → `quick`, T4 → `unspecified-high`, T5-T6 → `quick`, T7 → `deep`
- **Wave 2**: **5** — T8-T10,T12 → `deep`, T11 → `quick`
- **Wave 3**: **3** — T13,T15 → `deep`, T14 → `unspecified-high`
- **Wave 4**: **3** — T16 → `deep`, T17 → `quick`, T18 → `unspecified-high`
- **FINAL**: **4** — F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> 实现 + 测试 = 一个任务。不可拆分。
> 每个任务必须包含：推荐代理配置 + 并行化信息 + QA 场景。
> **缺少 QA 场景的任务不完整。无例外。**
> **格式**：任务标签必须使用裸数字：`1.`, `2.`, `3.` — 不可使用 `T1.`, `Task 1.`, `Phase 1:`。
> 最终验证波标签必须使用 `F1.`, `F2.` — 不可使用 `T-F1.`, `F-1.`, `Final 1.`。

- [x] 1. 项目脚手架：创建 deploy/CPP/ 目录结构和 CMakeLists.txt

  **What to do**:
  - 创建目录结构：`deploy/CPP/{include/, src/, tests/, cmake/}`
  - 编写顶层 `CMakeLists.txt`：设置 C++17, CUDA 11.3+, sm80/sm86/sm89, 查找 TensorRT/CUDA 包
  - 编写 `cmake/FindTensorRT.cmake`：定位 nvinfer 库和头文件
  - 集成 GoogleTest (FetchContent)
  - 启用 CTest，添加 cuda-memcheck 自定义 target
  - 配置 RPATH 确保运行时能找到 TensorRT .so 文件
  - 验证：`mkdir build && cd build && cmake .. && make -j` 成功

  **Must NOT do**:
  - 不要创建 setup.py 或 PyTorch Extension 风格的构建
  - 不要链接 openpoints/ 下的任何代码
  - 不要使用硬编码的 TensorRT 路径（使用 find_package/cmake 变量）

  **Recommended Agent Profile**:
  - **Category**: `quick` — 文件创建和 CMake 配置，模式固定
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 T2-T7 并行）
  - **Blocks**: T2-T18（所有后续 C++ 任务）
  - **Blocked By**: None

  **References**:
  - `openpoints/cpp/pointops/` — 现有 CMake CUDA 项目参考
  - `openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu` — CUDA kernel 线程块模式
  - `deploy/trt_utils.py:40-41` — TRT 库路径参考

  **Acceptance Criteria**:
  - [ ] `deploy/CPP/` 包含 CMakeLists.txt, cmake/FindTensorRT.cmake, include/, src/, tests/
  - [ ] `mkdir -p build && cd build && cmake ..` 成功检测到 TensorRT 和 CUDA
  - [ ] `make -j` 成功构建

  **QA Scenarios**:

  ```
  Scenario: CMake 配置和空项目构建
    Tool: Bash
    Preconditions: TensorRT 8.6 安装, CUDA 11.8 可用
    Steps:
      1. cd deploy/CPP && mkdir -p build && cd build
      2. cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1 | tee cmake_output.log
      3. grep -q "Found TensorRT" cmake_output.log
      4. grep -q "Found CUDA" cmake_output.log
      5. make -j$(nproc) 2>&1
    Expected Result: CMake 成功，make 退出码 0
    Failure Indicators: CMake FATAL_ERROR 或 make 编译失败
    Evidence: .omo/evidence/task-1-cmake-config.log

  Scenario: ctest 框架可运行（即使无测试）
    Tool: Bash
    Steps: ctest --output-on-failure
    Expected Result: ctest 运行成功（0 tests 也视为成功）
    Evidence: .omo/evidence/task-1-ctest.txt
  ```

  **Commit**: YES
  - Message: `build(deploy): add CMake/CUDA/TRT C++ project scaffolding in deploy/CPP`
  - Files: `deploy/CPP/CMakeLists.txt`, `deploy/CPP/cmake/FindTensorRT.cmake`

- [x] 2. Logger 工具类：实现 TrLogger + CUDA/TRT 错误检查宏

  **What to do**:
  - 实现 `TrLogger` 类（继承 `nvinfer1::ILogger`），重定向 TRT 日志到 stderr
  - 实现 `CHECK_CUDA(call)` 宏：检查 cudaError_t，失败时打印文件名/行号并 abort
  - 实现 `CHECK_TRT(call)` 宏：检查 TRT 状态，失败时输出错误信息
  - 所有 CUDA kernel 启动后调用 `cudaGetLastError()` + `cudaDeviceSynchronize()`

  **Must NOT do**:
  - 不要引入 spdlog 或其它日志库
  - 不要吞掉错误 — 所有错误必须传播

  **Recommended Agent Profile**:
  - **Category**: `quick` — 简单宏和虚函数实现
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 T1,T3-T7 并行）
  - **Blocks**: T14, T15
  - **Blocked By**: T1

  **References**:
  - `openpoints/cpp/pointnet2_batch/src/ball_query.cpp:14-26` — CHECK_CUDA 宏模式
  - TensorRT nvinfer1::ILogger 接口

  **Acceptance Criteria**:
  - [ ] `include/logger.h` 包含 TrLogger 类和 CHECK_CUDA/CHECK_TRT 宏
  - [ ] 编译通过

  **QA Scenarios**:

  ```
  Scenario: CHECK_CUDA 检测到错误并中止
    Tool: Bash
    Steps:
      1. 编写测试程序调用 cudaMalloc(nullptr, 1e12) 并用 CHECK_CUDA 包裹
      2. 编译运行，捕获 stderr 和退出码
    Expected Result: stderr 包含文件名和行号，程序以非零退出码终止
    Failure Indicators: 错误被吞掉或正常退出
    Evidence: .omo/evidence/task-2-check-cuda.txt
  ```

  **Commit**: YES（与 T1 同组）
  - Files: `include/logger.h`

- [x] 3. 通用类型定义：PointCloud, FeatureStats, Config, InferenceResult 结构体

  **What to do**:
  - `PointCloud`: coord(N×3), feat(N×3), label(N), num_points
  - `FeatureStats`: feat_mean[3], feat_std[3], z_mean, z_std
  - `Config`: engine_path, stats_path, data_dir, num_files, min_n(1024), max_n(30000), voxel_size(0.1), warmup_runs(5), output_path
  - `InferenceResult`: logits(N×2), predictions(N), latency_ms
  - 所有结构体使用构造函数初始化默认值

  **Must NOT do**:
  - 不要在头文件中实现业务逻辑
  - 不要使用继承或多态（保持 POD 风格）

  **Recommended Agent Profile**:
  - **Category**: `quick` — 简单 POD 结构体
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 T1-T2,T4-T7 并行）
  - **Blocks**: T4, T5, T7-T12, T17
  - **Blocked By**: T1

  **References**:
  - `deploy/common.py:53-74` — preprocess_test 数据结构
  - `deploy/common.py:77-85` — load_stats 返回格式
  - `deploy/trt_inference.py:148-172` — argparse 参数定义

  **Acceptance Criteria**:
  - [ ] `include/types.h` 包含所有结构体，编译通过

  **QA Scenarios**:

  ```
  Scenario: 结构体默认初始化和编译
    Tool: Bash
    Steps:
      1. 编写测试程序：实例化所有结构体，断言默认值
      2. 编译运行
    Expected Result: 编译成功，运行退出码 0
    Evidence: .omo/evidence/task-3-types.txt
  ```

  **Commit**: YES（与 T1 同组）
  - Files: `include/types.h`

- [x] 4. PLY 读取器：基于 tinyply 3.0 实现 PointCloud 加载

  **What to do**:
  - 将 tinyply 3.0 单头文件放入 `include/tinyply/tinyply.h`
  - 实现 `PlyReader::load(path) → PointCloud`
  - 支持 binary little-endian 和 ASCII 格式
  - 按名称提取字段：x, y, z → coord; rcs, snr, v → feat; label → label
  - 缺失字段报错退出；NaN 替换为 0.0
  - 与 Python plyfile 对比验证

  **Must NOT do**:
  - 不要引入 PCL 或其它重型依赖
  - 不要假设字段顺序 — 按名称匹配

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 第三方库集成，PLY 格式解析
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 T1-T3,T5-T7 并行）
  - **Blocks**: T9, T16
  - **Blocked By**: T1, T3

  **References**:
  - `deploy/common.py:24-50` — load_data_ply Python 实现
  - tinyply GitHub API: `tinyply::PlyFile`, `request_properties_from_element`

  **Acceptance Criteria**:
  - [ ] `include/ply_reader.h` + `src/ply_reader.cpp`
  - [ ] 读取已知 PLY 文件，数值与 Python 一致

  **QA Scenarios**:

  ```
  Scenario: 读取已知雷达 PLY 文件 — 字段正确性
    Tool: Bash
    Preconditions: data/RadarClassi/radarfull/raw/ 下有 .ply 文件
    Steps:
      1. C++ 程序加载文件，输出 coord 前 3 点和 feat 前 3 点
      2. Python plyfile 读取同一文件，输出相同内容
      3. diff 对比
    Expected Result: 数值完全一致（逐元素相等）
    Evidence: .omo/evidence/task-4-ply-read.txt

  Scenario: 缺失字段报错
    Tool: Bash
    Steps:
      1. 创建缺少 rcs 字段的测试 PLY
      2. 尝试加载，检查 stderr 和退出码
    Expected Result: 非零退出码，stderr 包含 "rcs"
    Evidence: .omo/evidence/task-4-ply-error.txt
  ```

  **Commit**: YES（与 T1 同组）
  - Files: `include/ply_reader.h`, `src/ply_reader.cpp`, `include/tinyply/tinyply.h`

- [x] 5. 特征统计 JSON 读取器 + pth→JSON Python 转换脚本

  **What to do**:
  - Python 脚本 `deploy/CPP/scripts/convert_stats.py`：读取 .pth → 输出 JSON
    - JSON 格式: `{"feat_mean": [f1,f2,f3], "feat_std": [s1,s2,s3], "z_mean": zm, "z_std": zs}`
  - C++ `StatsReader::load(path) → FeatureStats`：使用 nlohmann/json 或手写简易 JSON 解析器
  - 运行现有 stats 文件验证转换正确性

  **Must NOT do**:
  - 不要尝试在 C++ 中直接读取 .pth（避免引入 libtorch）
  - 不要使用复杂的 JSON 库 — nlohmann/json single-header 即可

  **Recommended Agent Profile**:
  - **Category**: `quick` — Python 脚本 + 简单 JSON 解析
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 T1-T4,T6-T7 并行）
  - **Blocks**: T10, T16
  - **Blocked By**: T1, T3

  **References**:
  - `deploy/common.py:77-85` — load_stats 格式
  - 现有 stats 文件路径：`data/RadarClassi/radarfull/processed/feat_stats_area5.pth`

  **Acceptance Criteria**:
  - [ ] `scripts/convert_stats.py` 运行成功，生成有效 JSON
  - [ ] C++ StatsReader 正确解析 JSON，数值与 .pth 一致

  **QA Scenarios**:

  ```
  Scenario: Python 转换脚本生成正确 JSON
    Tool: Bash
    Steps:
      1. python deploy/CPP/scripts/convert_stats.py --input data/RadarClassi/radarfull/processed/feat_stats_area5.pth --output /tmp/stats.json
      2. python -c "import json; d=json.load(open('/tmp/stats.json')); print(d['feat_mean'], d['z_mean'])"
    Expected Result: JSON 包含 4 个键，数值与 torch.load 原始值一致
    Evidence: .omo/evidence/task-5-convert.json

  Scenario: C++ 解析 JSON 正确
    Tool: Bash
    Steps:
      1. C++ 测试程序加载 /tmp/stats.json
      2. 对比 FeatureStats 字段值与 Python torch.load 输出
    Expected Result: feat_mean/std, z_mean/std 数值完全一致
    Evidence: .omo/evidence/task-5-stats-read.txt
  ```

  **Commit**: YES（与 T1 同组）
  - Files: `scripts/convert_stats.py`, `include/stats_reader.h`, `src/stats_reader.cpp`

- [x] 6. 黄金数据生成器：Python 脚本生成 CUDA kernel 测试向量

  **What to do**:
  - 创建 `deploy/CPP/scripts/gen_golden_data.py`
  - 生成 FNV hash 测试向量：随机 coord 数组 → 计算 FNV64-1A hash → 保存为二进制文件
  - 生成 voxelize 测试向量：已知 coord → voxelize → 保存 idx_sort, voxel_idx, count, idx_points
  - 生成 scatter_mean 测试向量：已知 logits + idx → scatter_mean → 保存结果
  - 所有测试数据同时保存为 .bin（C++ 读取）和 .npy（验证参考）
  - 使用固定随机种子确保可复现

  **Must NOT do**:
  - 不要依赖部署模型（测试向量使用随机小数据，无需加载模型权重）

  **Recommended Agent Profile**:
  - **Category**: `quick` — Python 数据处理脚本
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 T1-T5,T7 并行）
  - **Blocks**: T8, T9, T12
  - **Blocked By**: None（纯 Python，独立运行）

  **References**:
  - `openpoints/dataset/data_util.py:92-105` — FNV64-1A Python 实现
  - `openpoints/dataset/data_util.py:127` — voxelize Python 实现
  - `deploy/trt_inference.py:82-89` — scatter_mean 使用模式

  **Acceptance Criteria**:
  - [ ] `scripts/gen_golden_data.py` 运行成功
  - [ ] 生成 `tests/data/fnv_golden.bin`, `tests/data/voxel_golden.bin`, `tests/data/scatter_golden.bin`

  **QA Scenarios**:

  ```
  Scenario: 黄金数据生成且自洽
    Tool: Bash
    Steps:
      1. python deploy/CPP/scripts/gen_golden_data.py --output_dir deploy/CPP/tests/data
      2. ls deploy/CPP/tests/data/*.bin deploy/CPP/tests/data/*.npy
      3. python -c "import numpy; d=numpy.load('tests/data/fnv_golden.npy'); print(d.shape, d.dtype)"
    Expected Result: 所有 .bin 和 .npy 文件生成，形状和类型正确
    Evidence: .omo/evidence/task-6-golden-gen.txt
  ```

  **Commit**: YES（与 T1 同组）
  - Files: `scripts/gen_golden_data.py`

- [x] 7. 随机数工具：实现 numpy.random.seed(100) 兼容的 MT19937 PRNG

  **What to do**:
  - 实现 Mersenne Twister 19937 (MT19937) 随机数生成器
  - 匹陪 numpy 的初始化方式（seed=100 时状态完全一致）
  - 实现 `uniform_int(low, high)` — 用于 shuffle 索引
  - 实现 `shuffle(arr, n)` — Fisher-Yates 洗牌，与 numpy 顺序一致
  - 使用固定 seed 调用验证：生成 100 个随机数，对比 numpy 输出

  **Must NOT do**:
  - 不要使用 C++ `<random>` 的 mt19937（初始化方式与 numpy 不同）
  - 不要依赖 numpy C API

  **Recommended Agent Profile**:
  - **Category**: `deep` — 需要精确匹配 numpy MT19937 实现细节
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 T1-T6 并行）
  - **Blocks**: T9
  - **Blocked By**: T1, T3

  **References**:
  - numpy MT19937 源码参考：`numpy/random/mtrand/randomkit.c`
  - `deploy/common.py:53-74` — preprocess_test 中的 np.random.shuffle 调用

  **Acceptance Criteria**:
  - [ ] `include/random_util.h` + `src/random_util.cpp`
  - [ ] seed=100 时生成的前 20 个随机数与 numpy 完全一致

  **QA Scenarios**:

  ```
  Scenario: seed=100 输出与 numpy 完全一致
    Tool: Bash
    Steps:
      1. C++ 程序 seed=100, 调用 uniform_int(0, 9999) 50 次，输出
      2. Python: np.random.seed(100); [np.random.randint(0,10000) for _ in range(50)]
      3. diff 对比
    Expected Result: 50 个整数完全一致
    Evidence: .omo/evidence/task-7-mt19937.txt

  Scenario: Fisher-Yates shuffle 与 numpy 一致
    Tool: Bash
    Steps:
      1. C++ 程序创建 arr=[0..99], seed=100, shuffle, 输出前 20 个
      2. Python: np.random.seed(100); arr=np.arange(100); np.random.shuffle(arr); print(arr[:20])
      3. diff 对比
    Expected Result: 前 20 个元素完全一致
    Evidence: .omo/evidence/task-7-shuffle.txt
  ```

  **Commit**: YES（与 T1 同组）
  - Files: `include/random_util.h`, `src/random_util.cpp`

- [x] 8. FNV64-1A 哈希 CUDA kernel：实现与测试（对比黄金数据）

  **What to do**:
  - CUDA kernel: FNV64-1A 哈希每个点的体素坐标 (ix, iy, iz) → 64-bit hash
  - 公式: `hash = 14695981039346656037ULL`; `hash = (hash ^ byte) * 1099511628211ULL`
  - 体素坐标计算: `int32_t ix = floorf(coord_x / voxel_size)`; iy, iz 同理
  - 每个点处理: 使用 `__restrict__`, 1D grid + 1D block, THREADS_PER_BLOCK=256
  - 编写单元测试 `tests/test_fnv_hash.cu`：加载黄金数据 .bin，逐元素对比 hash 输出
  - 要求：所有体素 hash 值位精确一致（0 容差）

  **Must NOT do**:
  - 不要使用不同的 FNV 变体（必须 FNV-1A，不是 FNV-1）
  - 不要忽略 uint64 溢出语义（C++ 无符号整数溢出是定义良好的环绕行为）

  **Recommended Agent Profile**:
  - **Category**: `deep` — CUDA kernel 实现 + 位精确验证
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 T9-T12 并行）
  - **Blocks**: T9
  - **Blocked By**: T1, T3, T6

  **References**:
  - `openpoints/dataset/data_util.py:92-105` — Python FNV64-1A 实现（uint64 溢出语义）
  - `openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu:15-31` — CUDA kernel 1D 线程索引模式
  - `openpoints/cpp/pointnet2_batch/src/cuda_utils.h` — THREADS_PER_BLOCK, DIVUP 宏

  **Acceptance Criteria**:
  - [ ] `src/kernels/fnv_hash.cu` — CUDA kernel 实现
  - [ ] `tests/test_fnv_hash.cu` — GoogleTest 测试
  - [ ] `ctest -R fnv` → PASS（所有 hash 值位精确一致）

  **QA Scenarios**:

  ```
  Scenario: FNV hash 与 Python 位精确一致
    Tool: Bash
    Preconditions: T6 已生成 tests/data/fnv_golden.bin
    Steps:
      1. cd deploy/CPP/build && ctest -R fnv --output-on-failure -V 2>&1
      2. grep -q "PASSED" ctest_output.txt
    Expected Result: 测试通过，逐元素 hash 值完全一致
    Failure Indicators: 任何 hash 值不匹配
    Evidence: .omo/evidence/task-8-fnv-test.txt

  Scenario: 边界情况 — 负坐标和零坐标
    Tool: Bash
    Steps:
      1. 编写额外测试：coord = [-1.0, 0.0, 100.5], voxel_size=0.1
      2. 验证 floorf 行为与 numpy floor 一致
    Expected Result: hash 输出与 Python 一致
    Evidence: .omo/evidence/task-8-fnv-edge.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add FNV64-1A CUDA hash kernel with golden tests`
  - Files: `src/kernels/fnv_hash.cu`, `tests/test_fnv_hash.cu`

- [x] 9. 体素化 CUDA kernel：完整体素化 + 子云分割 + 测试

  **What to do**:
  - CUDA kernel 完整实现 `voxelize(coord, voxel_size, mode=1)`:
    - 步骤 1: FNV hash 每个点 → 调用 T8 的 kernel
    - 步骤 2: 按 hash 排序 (thrust::sort_by_key 或 cub::DeviceRadixSort)
    - 步骤 3: 计算每个体素的点数 (count)，获取 max_count
    - 步骤 4: 生成子云索引列表 idx_points (每个子云从每个体素取第 i 个点)
  - 实现 `Voxelizer` 类：封装完整 pipeline，返回 `std::vector<std::vector<int>>` idx_points
  - 编写单元测试 `tests/test_voxelize.cu`：使用黄金数据对比 idx_points 列表
  - 允许排序后索引因 stable sort 实现差异导致的非确定性（但 count 必须完全一致）

  **Must NOT do**:
  - 不要在 GPU 上执行串行循环
  - 不要假设 voxel_size 是常量（从 Config 读取）

  **Recommended Agent Profile**:
  - **Category**: `deep` — 复杂 CUDA pipeline (hash→sort→count→split)
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 T8,T10-T12 并行）
  - **Blocks**: T16
  - **Blocked By**: T1, T3, T4, T6, T7, T8

  **References**:
  - `openpoints/dataset/data_util.py:101-157` — voxelize Python 完整实现（NV hash, unique, argsort, count, 子云分割）
  - `deploy/common.py:53-74` — preprocess_test 使用 voxelize 的方式
  - `openpoints/cpp/pointops/src/grouping/grouping_cuda_kernel.cu` — 排序后分组模式参考

  **Acceptance Criteria**:
  - [ ] `include/voxelizer.h` + `src/voxelizer.cpp` + `src/kernels/voxelize.cu`
  - [ ] `tests/test_voxelize.cu` — GoogleTest 测试
  - [ ] 子云索引列表与 Python 输出结构一致（每个子云点数相同，索引集合等价）

  **QA Scenarios**:

  ```
  Scenario: 体素化输出与 Python 一致
    Tool: Bash
    Preconditions: T6 已生成 tests/data/voxel_golden.bin
    Steps:
      1. cd deploy/CPP/build && ctest -R voxelize --output-on-failure -V 2>&1
    Expected Result: 每个子云的 count 完全一致，idx_points 作为集合等价
    Failure Indicators: count 不一致或子云数量不同
    Evidence: .omo/evidence/task-9-voxelize-test.txt

  Scenario: 空输入或单点输入
    Tool: Bash
    Steps:
      1. 测试 N=1 点输入：应产生 1 个子云，子云包含该唯一点
      2. 测试 N=0 点输入：应产生空 idx_points 列表
    Expected Result: 不崩溃，输出合理
    Evidence: .omo/evidence/task-9-voxelize-edge.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add CUDA voxelize kernel with sub-cloud splitting`
  - Files: `include/voxelizer.h`, `src/voxelizer.cpp`, `src/kernels/voxelize.cu`, `tests/test_voxelize.cu`

- [x] 10. 特征归一化 + 子云预处理 CUDA kernel：实现与测试

  **What to do**:
  - 实现 `preprocess_subcloud` C++ 等效函数（CPU + 可选 CUDA 加速）:
    - `coord_part -= coord_part.min(0)` — 平移至原点
    - `pos -= pos.mean(dim=0)` — 中心化
    - `pos[:, gravity_dim(2)] -= pos[:, gravity_dim].min()` — z 轴归零
    - `feat = (feat - feat_mean) / (feat_std + 1e-5)` — 特征归一化
    - `height = (height - z_mean) / (z_std + 1e-5)` — 高度归一化
    - `x = concat([feat, height])` → (1, 4, N)
    - `pos` → (1, N, 3)
  - 输出为已分配 GPU 内存的 CudaBuffer（float32）
  - 编写测试：使用 T5 的统计文件 + 已知子云输入，对比 Python 输出

  **Must NOT do**:
  - 不要修改原始输入数据（使用 copy）
  - 不要忘记 clamp(min=1e-5) 防止除零

  **Recommended Agent Profile**:
  - **Category**: `deep` — 数值敏感的计算，需精确匹配 Python 数学行为
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 T8-T9,T11-T12 并行）
  - **Blocks**: T16
  - **Blocked By**: T1, T3, T5

  **References**:
  - `deploy/common.py:88-115` — preprocess_subcloud Python 完整实现
  - `openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu:15-30` — CUDA kernel 指针偏移模式

  **Acceptance Criteria**:
  - [ ] `include/preprocessor.h` + `src/preprocessor.cpp`
  - [ ] 归一化后 pos 和 x 数值与 Python 一致（容差 1e-5）
  - [ ] GPU 输出可直接传递给 TRT（无需额外格式转换）

  **QA Scenarios**:

  ```
  Scenario: 归一化输出与 Python 一致
    Tool: Bash
    Steps:
      1. 使用相同 stats 和子云数据运行 C++ 和 Python
      2. 对比 pos 和 x 的数值（逐元素最大绝对误差）
    Expected Result: 最大绝对误差 < 1e-5
    Evidence: .omo/evidence/task-10-preprocess.txt

  Scenario: 除零保护 — std=0
    Tool: Bash
    Steps:
      1. 使用 feat_std = [0.0, 0.0, 0.0] 输入
      2. 验证不产生 inf 或 nan
    Expected Result: feat 归一化为 0（因 clamp(min=1e-5)）
    Evidence: .omo/evidence/task-10-divzero.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add feature normalization and sub-cloud preprocessor`
  - Files: `include/preprocessor.h`, `src/preprocessor.cpp`

- [x] 11. 子云填充和分块逻辑：padding + chunking 实现与测试

  **What to do**:
  - 实现 `pad_subcloud(pos, x, min_n)` — 复制最后一点填充至 min_n
  - 实现 `split_oversized_subcloud(pos, x, max_n)` — 将 >max_n 的子云分割为多个 chunk
  - 实现 `trim_padding(logits, N_true, N_padded)` — 裁剪推理输出
  - 边界情况处理: N < min_n (padding), min_n ≤ N ≤ max_n (不变), N > max_n (chunk → 多块)
  - 编写测试覆盖所有边界情况

  **Must NOT do**:
  - 不要使用动态内存分配在 kernel 循环中
  - 不要假设 min_n 和 max_n 是编译时常量

  **Recommended Agent Profile**:
  - **Category**: `quick` — CPU 端辅助逻辑，非性能关键路径
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 T8-T10,T12 并行）
  - **Blocks**: T16
  - **Blocked By**: T1, T3

  **References**:
  - `deploy/trt_inference.py:40-52` — pad_subcloud Python 实现
  - `deploy/trt_build.py:134` — max_n=30000 参数

  **Acceptance Criteria**:
  - [ ] `include/subcloud_utils.h` + `src/subcloud_utils.cpp`
  - [ ] N=500 → 填充至 1024, 输出 N_padded=1024
  - [ ] N=50000 → 分割为 [N_1, N_2] 两块, 每块 ≤ max_n
  - [ ] padding 后 填充部分=最后一点

  **QA Scenarios**:

  ```
  Scenario: N < min_n 填充正确
    Tool: Bash
    Steps:
      1. 输入 pos=(1,500,3), x=(1,4,500), min_n=1024
      2. 验证输出 shape=(1,1024,3) 和 (1,4,1024)
      3. 验证 pos[0,500:1024] 全等于 pos[0,499]
    Expected Result: 填充部分点数正确，值为最后一点复制
    Evidence: .omo/evidence/task-11-pad.txt

  Scenario: N > max_n 分块正确
    Tool: Bash
    Steps:
      1. 输入 pos=(1,50000,3), max_n=30000
      2. 验证输出两块 shape: (1,25000,3) 和 (1,25000,3)（或类似分割）
      3. 验证两块拼接后 = 原始（忽略填充）
    Expected Result: 两块 拼接后可恢复原始数据
    Evidence: .omo/evidence/task-11-chunk.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add sub-cloud padding, chunking, and trimming utilities`
  - Files: `include/subcloud_utils.h`, `src/subcloud_utils.cpp`

- [x] 12. Scatter-mean CUDA kernel：实现与测试（对比黄金数据）

  **What to do**:
  - CUDA kernel: 实现 `scatter_mean(input, index, dim=0, reduce='mean')`:
    - 并行策略：每个输出位置一个线程 block，或使用原子操作
    - 阶段 1: atomicAdd 累加 input 到 output（float）
    - 阶段 2: atomicAdd 累加计数到 count（int）
    - 阶段 3: output[i] /= count[i]（除法）
  - 封装为 `ScatterMean` 类：接受 GPU 内存指针，返回结果
  - 编写测试 `tests/test_scatter_mean.cu`：使用黄金数据对比输出
  - 容差：1e-4（因 atomicAdd 浮点累加顺序可能不同）

  **Must NOT do**:
  - 不要在 CPU 上执行 scatter（必须 GPU kernel）
  - 不要假设指数范围（idx 可以是任意 int64 值）

  **Recommended Agent Profile**:
  - **Category**: `deep` — CUDA atomicAdd reduce kernel + 精度权衡
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 T8-T11 并行）
  - **Blocks**: T16
  - **Blocked By**: T1, T3, T6

  **References**:
  - `deploy/trt_inference.py:82-89` — scatter_mean Python 使用（dim=0, reduce='mean'）
  - `openpoints/cpp/pointops/src/aggregation/aggregation_cuda_kernel.cu` — atomicAdd CUDA 模式参考
  - torch_scatter 文档：scatter(src, index, dim, reduce)

  **Acceptance Criteria**:
  - [ ] `include/scatter_mean.h` + `src/kernels/scatter_mean.cu`
  - [ ] `tests/test_scatter_mean.cu` — GoogleTest 测试
  - [ ] 逐元素最大绝对误差 < 1e-4

  **QA Scenarios**:

  ```
  Scenario: scatter_mean 输出在容差范围内
    Tool: Bash
    Preconditions: T6 已生成 tests/data/scatter_golden.bin
    Steps:
      1. cd deploy/CPP/build && ctest -R scatter --output-on-failure -V 2>&1
    Expected Result: 测试通过，所有输出点在 1e-4 容差内
    Evidence: .omo/evidence/task-12-scatter-test.txt

  Scenario: 大指数范围和空桶
    Tool: Bash
    Steps:
      1. 测试 index 包含 [0, 100000, 200000] 的稀疏场景
      2. 验证空桶（无 input 的 index）输出为 0
    Expected Result: 非空桶正确平均，空桶为 0，无越界
    Evidence: .omo/evidence/task-12-scatter-sparse.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add scatter-mean CUDA kernel with atomicAdd reduce`
  - Files: `include/scatter_mean.h`, `src/kernels/scatter_mean.cu`, `tests/test_scatter_mean.cu`

- [x] 13. GPU 内存 RAII 管理器：CudaBuffer, CudaStream, DeviceGuard

  **What to do**:
  - `CudaBuffer` 类：RAII 封装 cudaMalloc/cudaFree
    - 构造函数接受 size (bytes) 或 shape + dtype
    - 移动语义（禁止拷贝）
    - `data_ptr()` 返回 void*
    - `upload(host_ptr, size)` — cudaMemcpy H→D
    - `download(host_ptr, size)` — cudaMemcpy D→H
    - `memset(value)` — cudaMemset
  - `CudaStream` 类：RAII 封装 cudaStreamCreate/Destroy
    - `synchronize()` — cudaStreamSynchronize
    - `native()` — 返回原始 cudaStream_t（供 TRT 使用）
  - `DeviceGuard` 类：RAII 封装 cudaSetDevice（构造时切换到指定 device，析构时恢复）

  **Must NOT do**:
  - 不要使用 raw cudaMalloc/cudaFree 在业务代码中（全部使用 CudaBuffer）
  - 不要忘记处理 cudaMalloc 返回的 OOM

  **Recommended Agent Profile**:
  - **Category**: `deep` — RAII 模式 + CUDA 错误处理 + 移动语义
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 T14 并行）
  - **Parallel Group**: Wave 3（与 T14 并行；T15 需等待二者）
  - **Blocks**: T15, T16
  - **Blocked By**: T1

  **References**:
  - `openpoints/cpp/pointnet2_batch/src/ball_query.cpp:22-26` — CHECK 宏模式（可借鉴用于 CudaBuffer 的错误检查）
  - TensorRT IExecutionContext::enqueueV3 文档 — 需要 cudaStream_t 参数
  - C++11 移动语义参考：Rule of Five

  **Acceptance Criteria**:
  - [ ] `include/cuda_utils.h` — CudaBuffer, CudaStream, DeviceGuard
  - [ ] CudaBuffer 移动构造后原对象 data_ptr() == nullptr
  - [ ] CudaStream::native() 返回有效 cudaStream_t
  - [ ] DeviceGuard 析构后恢复原始 device

  **QA Scenarios**:

  ```
  Scenario: CudaBuffer 上传下载正确
    Tool: Bash
    Steps:
      1. 创建 host 数组 float[1024] = {1..1024}
      2. CudaBuffer::upload → GPU → CudaBuffer::download → host2
      3. 逐元素对比 host 和 host2
    Expected Result: 完全一致
    Evidence: .omo/evidence/task-13-cudabuffer.txt

  Scenario: CudaBuffer OOM 抛出异常
    Tool: Bash
    Steps:
      1. 尝试 cudaMalloc(1e15) 通过 CudaBuffer
      2. 捕获异常
    Expected Result: 抛出 std::runtime_error，不崩溃
    Evidence: .omo/evidence/task-13-oom.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add RAII CUDA memory manager (CudaBuffer, CudaStream, DeviceGuard)`
  - Files: `include/cuda_utils.h`

- [x] 14. TRT 引擎加载器：反序列化 .engine 文件 + I/O 张量信息查询

  **What to do**:
  - 实现 `TrEngine` 类：
    - 构造函数接受 .engine 文件路径 + TrLogger 引用
    - `nvinfer1::IRuntime::deserializeCudaEngine()` 反序列化
    - 查询 I/O 张量信息：名称、形状、数据类型（nvinfer1::ITensor）
    - 验证输入形状符合预期：pos (1, -1, 3), x (1, 4, -1)，output (1, 2, -1)
    - 输出引擎信息：I/O 名称列表，数据类型，最大批大小
  - 错误处理：文件不存在、反序列化失败、格式不兼容

  **Must NOT do**:
  - 不要创建 nvonnxparser（不需要 ONNX 解析）
  - 不要在引擎加载后修改网络结构

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — TensorRT C++ API 集成，需要正确使用 nvinfer1 接口
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 T13 并行）
  - **Parallel Group**: Wave 3（与 T13 并行；T15 需等待二者）
  - **Blocks**: T15, T16
  - **Blocked By**: T1, T2

  **References**:
  - `deploy/trt_utils.py:70-113` — TRTSession.__init__ Python 实现（数值等价）
  - TensorRT 8.6 C++ API: `nvinfer1::createInferRuntime()`, `nvinfer1::ICudaEngine`
  - `deploy/trt_build.py:79-88` — 动态形状定义（min/opt/max 参考）

  **Acceptance Criteria**:
  - [ ] `include/trt_engine.h` + `src/trt_engine.cpp`
  - [ ] 加载 `deploy/trt_model_fp32.engine` 成功
  - [ ] 查询输入 "pos" 形状: (-1, -1, 3), "x" 形状: (-1, 4, -1)
  - [ ] 查询输出形状: (-1, 2, -1)

  **QA Scenarios**:

  ```
  Scenario: 加载 FP32 引擎并查询 I/O 信息
    Tool: Bash
    Preconditions: deploy/trt_model_fp32.engine 存在
    Steps:
      1. C++ 程序加载 engine，打印所有 I/O 张量名称、形状、dtype
      2. 对比 Python TRTSession.__repr__() 输出
    Expected Result: 张量名称、形状、dtype 与 Python 一致
    Evidence: .omo/evidence/task-14-engine-info.txt

  Scenario: 加载不存在文件报错
    Tool: Bash
    Steps:
      1. 尝试加载 /nonexistent.engine
    Expected Result: 抛出异常，stderr 包含文件路径
    Evidence: .omo/evidence/task-14-notfound.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add TRT engine loader (deserialize .engine file)`
  - Files: `include/trt_engine.h`, `src/trt_engine.cpp`

- [x] 15. TRT 推理执行器：ExecutionContext + 动态形状 + enqueueV3

  **What to do**:
  - 实现 `TrInference` 类（依赖 TrEngine）:
    - 创建 `nvinfer1::IExecutionContext`
    - `setInputShape(name, dims)` — 设置动态输入维度
    - `setTensorAddress(name, ptr)` — 绑定 GPU 内存指针
    - `enqueueV3(stream)` — 异步执行推理
    - CUDA stream 同步：调用 cudaStreamSynchronize
    - 支持 FP32 和 FP16 输入/输出（根据引擎 I/O 张量的 dtype 自动选择）
  - 封装 `run(pos_gpu_ptr, x_gpu_ptr) → output_gpu_ptr` 便捷方法
  - 测试：用随机数据运行推理，验证输出形状和数值范围

  **Must NOT do**:
  - 不要使用已废弃的 enqueue/enqueueV2 API
  - 不要在推理循环中频繁创建/销毁 context

  **Recommended Agent Profile**:
  - **Category**: `deep` — TensorRT 核心推理 API + 动态形状处理 + 异步执行
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: NO（串行依赖 T13, T14）
  - **Parallel Group**: Wave 3（T13, T14 完成后执行）
  - **Blocks**: T16, T18
  - **Blocked By**: T1, T2, T13, T14

  **References**:
  - `deploy/trt_utils.py:118-165` — TRTSession.run() Python 实现（set_input_shape, set_tensor_address, execute_async_v3, stream.synchronize）
  - TensorRT 8.6 C++ API: `nvinfer1::IExecutionContext::setInputShape()`, `setTensorAddress()`, `enqueueV3()`
  - `deploy/trt_build.py:79-88` — 动态形状配置（min_n/opt_n/max_n 参考值）

  **Acceptance Criteria**:
  - [ ] `include/trt_inference.h` + `src/trt_inference.cpp`
  - [ ] 随机输入 pos(1,1024,3) + x(1,4,1024) 推理成功
  - [ ] 输出形状 = (1, 2, 1024)，float32
  - [ ] 输出无 NaN 或 Inf

  **QA Scenarios**:

  ```
  Scenario: 随机输入推理成功且输出形状正确
    Tool: Bash
    Steps:
      1. 创建随机 pos(1,1024,3), x(1,4,1024) 上传到 GPU
      2. run() 推理
      3. 验证输出形状 (1,2,1024), 类型 float32
      4. 验证输出无 NaN/Inf
    Expected Result: 推理成功，输出有效
    Evidence: .omo/evidence/task-15-inference-shape.txt

  Scenario: 不同点数推理（动态形状）
    Tool: Bash
    Steps:
      1. 依次用 N=500, N=1024, N=4096, N=30000 运行推理
      2. 验证每次输出形状 (1,2,N)
    Expected Result: 所有 N 值推理成功，输出形状匹配
    Evidence: .omo/evidence/task-15-dynamic-shape.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add TRT inference runner with dynamic shapes and enqueueV3`
  - Files: `include/trt_inference.h`, `src/trt_inference.cpp`

- [x] 16. 主推理流水线：组装所有模块的完整 pipeline

  **What to do**:
  - 实现 `InferencePipeline` 类：
    - 初始化：加载 engine, stats, 创建 CudaStream
    - `process_file(ply_path) → InferenceResult`:
      1. PlyReader::load → PointCloud
      2. Voxelizer::voxelize → idx_points
      3. for each subcloud:
         a. Preprocessor::preprocess_subcloud → GPU buffers
         b. SubcloudUtils::pad→chunk→(if needed)
         c. TrInference::run → logits
         d. SubcloudUtils::trim_padding
      4. ScatterMean::reduce → merged logits (N, 2)
      5. argmax → predictions (N)
      6. 返回 InferenceResult
    - `process_directory(data_dir) → vector<InferenceResult>`
    - 与 Python `infer_one_cloud_trt` 对比输出

  **Must NOT do**:
  - 不要在推理循环中重新加载 engine 或 stats
  - 不要忘记 GPU/CPU 同步边界

  **Recommended Agent Profile**:
  - **Category**: `deep` — 多模块编排 + 错误传播 + GPU/CPU 同步
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 T17, T18 并行）
  - **Parallel Group**: Wave 4（与 T17, T18 并行）
  - **Blocks**: T17, T18
  - **Blocked By**: T4, T5, T9, T10, T11, T12, T13, T15

  **References**:
  - `deploy/trt_inference.py:59-89` — infer_one_cloud_trt 完整 Python 实现（subcloud 循环 + scatter merge）
  - `deploy/trt_inference.py:249-296` — 主循环（文件遍历 + 计时 + 输出）
  - `deploy/common.py` — 所有预处理函数

  **Acceptance Criteria**:
  - [ ] `include/pipeline.h` + `src/pipeline.cpp`
  - [ ] process_file 成功处理一个 PLY
  - [ ] 输出 predictions 形状与标签一致
  - [ ] 无 GPU 内存泄漏（cuda-memcheck 验证）

  **QA Scenarios**:

  ```
  Scenario: 端到端 — 单文件推理 vs Python 输出
    Tool: Bash
    Steps:
      1. C++ pipeline 处理 data/RadarClassi/radarfull/raw/ 首个 test PLY
      2. 保存 logits 为 .bin 文件
      3. Python 加载同一 PLY 运行 infer_one_cloud_trt
      4. Python 加载 C++ logits.bin，逐元素对比
    Expected Result: 逐元素最大绝对误差 < 1e-4
    Evidence: .omo/evidence/task-16-e2e-single.txt

  Scenario: GPU 内存安全
    Tool: Bash
    Steps:
      1. cuda-memcheck ./hpenet_trt_infer --engine deploy/trt_model_fp32.engine --data_dir /path --num_files 1
    Expected Result: cuda-memcheck 报告 0 errors
    Evidence: .omo/evidence/task-16-memcheck.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add full inference pipeline (PLY→voxelize→TRT→scatter→output)`
  - Files: `include/pipeline.h`, `src/pipeline.cpp`

- [x] 17. CLI 参数解析器 + main 入口

  **What to do**:
  - 实现 CLI 参数解析（不使用外部库，手写或使用 cxxopts/tclap header-only 库）:
    - `--engine` (默认: deploy/trt_model_fp32.engine)
    - `--stats` (默认: deploy/stats.json)
    - `--data_dir` (必需)
    - `--num_files` (默认: -1 全部)
    - `--min_n` (默认: 1024)
    - `--max_n` (默认: 30000)
    - `--voxel_size` (默认: 0.1)
    - `--warmup` (默认: 5)
    - `--output` (输出目录，默认: ./output)
    - `--benchmark` (开启性能基准模式)
    - `--seed` (随机种子，默认: 100)
  - 实现 `main()`:
    - 解析参数 → 构建 Config
    - 初始化 InferencePipeline
    - Warmup (min_n 大小的随机输入跑 N 次)
    - 遍历数据目录
    - 输出预测结果和准确率
    - 如果 --benchmark，输出延迟统计
  - 对照 Python argparse 接口保持兼容

  **Must NOT do**:
  - 不要引入 boost::program_options（避免重型依赖）
  - 不要硬编码路径

  **Recommended Agent Profile**:
  - **Category**: `quick` — CLI 参数解析 + main 函数编排
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 T18 并行）
  - **Parallel Group**: Wave 4（与 T16, T18 并行 — T17 依赖 T16 的实现，但 main.cpp 只需要 header）
  - **Blocks**: None（末端任务）
  - **Blocked By**: T3, T16

  **References**:
  - `deploy/trt_inference.py:148-172` — argparse 参数定义（所有参数名和默认值）
  - `deploy/trt_inference.py:186-191` — warmup 逻辑
  - cxxopts GitHub: https://github.com/jarro2783/cxxopts — header-only CLI 解析库

  **Acceptance Criteria**:
  - [ ] `src/main.cpp` — 包含 main 函数和参数解析
  - [ ] `make hpenet_trt_infer` 成功构建可执行文件
  - [ ] `./hpenet_trt_infer --help` 输出所有参数说明

  **QA Scenarios**:

  ```
  Scenario: --help 输出所有参数
    Tool: Bash
    Steps: ./hpenet_trt_infer --help 2>&1
    Expected Result: 输出包含 engine, stats, data_dir, num_files, min_n, max_n, voxel_size, warmup, benchmark
    Evidence: .omo/evidence/task-17-help.txt

  Scenario: 未知参数报错
    Tool: Bash
    Steps: ./hpenet_trt_infer --invalid_arg 2>&1 ; echo $?
    Expected Result: 非零退出码，stderr 包含错误说明
    Evidence: .omo/evidence/task-17-unknown-arg.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add CLI argument parser and main entry point`
  - Files: `src/main.cpp`

- [x] 18. 性能基准模块：延迟/吞吐量测量 + 与 Python 对比

  **What to do**:
  - 实现 `Benchmark` 类：
    - 使用 `cudaEventCreate/cudaEventRecord/cudaEventElapsedTime` 精确测量 GPU 时间
    - 测量指标：总延迟(ms)、纯推理延迟(ms)、预处理延迟(ms)、后处理延迟(ms)、吞吐量(files/sec)
    - 统计：min, max, mean, median, p50, p95, p99
  - 集成到 pipeline：每个阶段插入 CUDA event
  - 输出格式：CSV 表格或格式化文本
  - 对比：与 Python trt_inference.py 输出比较延迟

  **Must NOT do**:
  - 不要使用 std::chrono 测量 GPU 时间（只用于 CPU 时间，GPU 时间用 CUDA events）
  - 不要在 benchmark 中包含首次 warmup

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — CUDA event 计时 + 统计分析
  - **Skills**: (none needed)

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 T17 并行）
  - **Parallel Group**: Wave 4（与 T16, T17 并行）
  - **Blocks**: None（末端任务）
  - **Blocked By**: T15, T16

  **References**:
  - `deploy/trt_inference.py:245-296` — 计时和准确率统计（trt_time, trt_accs）
  - CUDA Programming Guide: cudaEvent_t, cudaEventCreate, cudaEventRecord, cudaEventElapsedTime
  - `deploy/trt_inference.py:186-191` — warmup 实现（5 次随机输入）

  **Acceptance Criteria**:
  - [ ] `include/benchmark.h` + `src/benchmark.cpp`
  - [ ] --benchmark 模式输出：latency(min/max/mean/p95), throughput
  - [ ] 延迟误差 < 0.1ms（基于 CUDA event 精度）
  - [ ] C++ 推理延迟 ≤ Python 的 80%

  **QA Scenarios**:

  ```
  Scenario: Benchmark 输出统计信息
    Tool: Bash
    Steps:
      1. ./hpenet_trt_infer --benchmark --num_files 3 --engine deploy/trt_model_fp32.engine 2>&1
      2. grep 输出中的 "Latency" "Throughput" 行
    Expected Result: 输出包含 min/max/mean/p50/p95/p99 延迟和吞吐量
    Evidence: .omo/evidence/task-18-benchmark.txt

  Scenario: C++ 延迟不高于 Python 的 80%
    Tool: Bash
    Steps:
      1. 运行 C++ benchmark (10 files)
      2. 运行 Python trt_inference.py (相同 10 files)
      3. 对比 mean latencys
    Expected Result: C++ mean latency ≤ Python mean latency × 0.8
    Evidence: .omo/evidence/task-18-vs-python.txt
  ```

  **Commit**: YES
  - Message: `feat(deploy/cpp): add CUDA event-based performance benchmark module`
  - Files: `include/benchmark.h`, `src/benchmark.cpp`

---

## Final Verification Wave (MANDATORY — 所有实现任务完成后)

> 4 个审查代理并行运行。全部必须 APPROVE。将合并结果呈现给用户，获得明确确认。
> **不要自动通过验证。等待用户明确批准后再标记工作完成。**
> **用户批准前切勿标记 F1-F4 为完成。**

- [x] F1. **方案合规审查** — `oracle` — **APPROVED** (Must Have 12/12 | Must NOT Have 5/5 | Tasks 18/18)
- [x] F2. **代码质量审查** — `unspecified-high` — **APPROVED** (Build PASS | Tests 3/3 | Files 30 clean)
- [x] F3. **实地手动 QA** — `unspecified-high` — **APPROVED** (Scenarios 5/5 | FP32推理 81.56% | 无占位桩)
- [x] F4. **范围保真度检查** — `deep` — **APPROVED** (Tasks 18/18 | Contamination CLEAN | Unaccounted CLEAN)

---

## Commit Strategy

| Wave | Tasks | Commit Message | Files Pattern |
|------|-------|----------------|---------------|
| 1 | T1-T7 | `build(deploy/cpp): add CMake project scaffolding, types, logger, PLY reader, stats reader, golden data generator, random util` | `deploy/CPP/{CMakeLists.txt,cmake/**,include/**,src/**,scripts/**,tests/data/**}` |
| 2 | T8 | `feat(deploy/cpp): add FNV64-1A CUDA hash kernel with golden tests` | `src/kernels/fnv_hash.cu, tests/test_fnv_hash.cu` |
| 2 | T9 | `feat(deploy/cpp): add CUDA voxelize kernel with sub-cloud splitting` | `include/voxelizer.h, src/voxelizer.cpp, src/kernels/voxelize.cu, tests/test_voxelize.cu` |
| 2 | T10-T12 | `feat(deploy/cpp): add preprocessor, subcloud utils, and scatter-mean CUDA kernel` | `include/preprocessor.h, src/preprocessor.cpp, include/subcloud_utils.h, src/subcloud_utils.cpp, include/scatter_mean.h, src/kernels/scatter_mean.cu, tests/test_scatter_mean.cu` |
| 3 | T13-T14 | `feat(deploy/cpp): add CUDA memory manager and TRT engine loader` | `include/cuda_utils.h, include/trt_engine.h, src/trt_engine.cpp` |
| 3 | T15 | `feat(deploy/cpp): add TRT inference runner with dynamic shapes` | `include/trt_inference.h, src/trt_inference.cpp` |
| 4 | T16-T18 | `feat(deploy/cpp): add inference pipeline, CLI, and benchmark module` | `include/pipeline.h, src/pipeline.cpp, src/main.cpp, include/benchmark.h, src/benchmark.cpp` |

---

## Success Criteria

### Verification Commands
```bash
# 构建
cd deploy/CPP && mkdir -p build && cd build
cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6
make -j$(nproc)

# 单元测试
ctest --output-on-failure

# GPU 内存检查
cuda-memcheck ./hpenet_trt_infer \
  --engine ../../deploy/trt_model_fp32.engine \
  --stats ../stats.json \
  --data_dir /path/to/radar/test \
  --num_files 3

# 性能基准
./hpenet_trt_infer --benchmark --num_files 10 --engine ../../deploy/trt_model_fp32.engine

# FP16 引擎测试
./hpenet_trt_infer --engine ../../deploy/trt_model_fp16.engine --num_files 3
```

### Final Checklist
- [ ] 全部 "Must Have" 已实现
- [ ] 全部 "Must NOT Have" 未违反
- [ ] 端到端 logits 误差 < 1e-4
- [ ] 分类结果 100% 一致
- [ ] cuda-memcheck 零错误
- [ ] C++ 延迟 ≤ Python 的 80%
- [ ] FP32 和 FP16 engine 均已验证
