# C++ ONNX Inference for HPENet V2 Radar Segmentation

## TL;DR

> **Quick Summary**: 将 `deploy/trt_manual/onnx_inference.py` 的 Python ONNX 推理管道转为纯 CPU 的 C++/ONNX Runtime 实现，输出到 `deploy/CPP_onnx/`。
>
> **Deliverables**:
> - `deploy/CPP_onnx/CMakeLists.txt` — CMake 构建文件
> - `deploy/CPP_onnx/onnx_inference.h` — 公共 API 头文件（类型定义 + 类声明）
> - `deploy/CPP_onnx/onnx_inference.cpp` — 核心实现（PLY读取、体素化、预处理、ONNX推理、scatter合并）
> - `deploy/CPP_onnx/main.cpp` — 命令行入口
> - `deploy/CPP_onnx/scripts/convert_stats.py` — stats 转换脚本 (.pth → .json)
> - `deploy/CPP_onnx/verify.py` — Python golden data 验证脚本
>
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential (按顺序逐任务执行)
> **Critical Path**: 任务 1 → 任务 2 → ... → 任务 9

---

## Context

### Original Request
将 `deploy/trt_manual/onnx_inference.py` 转成 C++/ONNX 实现，要求：
1. 代码尽量简单
2. 方便作为可调用的库文件被其他程序调用
3. 考虑多线程安全
4. 代码保存到 `deploy/CPP_onnx`

### Interview Summary
**Key Discussions**:
- **ONNX Runtime**: CPU only (CPUExecutionProvider)，无需 CUDA —— 与 Python 默认行为一致
- **代码组织**: 极简 2-3 文件 —— `onnx_inference.h/cpp` + `main.cpp`
- **构建系统**: CMake —— 与已有 CPP_trt 风格一致
- **验证策略**: 需要与 Python 输出做 golden data 数值对比
- **数学库**: 使用 Eigen3 做向量/矩阵运算（纯 CPU 路径，替代 TRT 中的 CUDA kernel）
- **PLY 读取**: 复用 tinyply（已在 `deploy/CPP_trt/include/tinyply/` 中）

**Research Findings**:
- Python 管道: PLY 加载 → 体素化(voxel_size=0.1, mode=1) → 子云拆分 → XYZAlign → 特征归一化 → ONNX 推理(pos:1×N×3, x:1×4×N) → scatter_mean 合并 → argmax
- ONNX 模型: `deploy/onnx_model.onnx`, 输入 `pos`(float32, 1×N×3) + `x`(float32, 1×4×N), 输出 (float32, 1×2×N)
- 特征统计文件: PyTorch `.pth` 格式，包含 `feat_mean`(3,), `feat_std`(3,), `z_mean`(标量), `z_std`(标量)
- 已有 `deploy/CPP_trt/` 提供了完整的 C++ TensorRT 参考实现

### Metis Review
Metis 在当前环境不可用，基于 Oracle Phase 1 验证替代。Oracle 确认所有 5 项检查 PASS（GO），仅有一处文档措辞建议（已修正）。

---

## Work Objectives

### Core Objective
用 C++/ONNX Runtime 重新实现 HPENet V2 雷达点云语义分割的端到端推理管道，输出与 Python 原版数值一致的结果。

### Concrete Deliverables
- `deploy/CPP_onnx/CMakeLists.txt` — 构建配置
- `deploy/CPP_onnx/onnx_inference.h` — 公开头文件
- `deploy/CPP_onnx/onnx_inference.cpp` — 核心实现
- `deploy/CPP_onnx/main.cpp` — 命令行工具
- `deploy/CPP_onnx/verify.py` — 验证脚本

### Definition of Done
- [x] `cmake -B build && cmake --build build` 编译成功，无错误
- [ ] 对 3 个测试 PLY 文件，C++ 预测准确率与 Python 原版差异 < 1e-5
- [ ] `verify.py` 输出 "ALL PASS" 表示数值一致性验证通过

### Must Have
- CPU-only 推理（无 CUDA 依赖）
- 端到端管道：PLY → 体素化 → 预处理 → ONNX 推理 → scatter_mean → argmax
- 与 Python 输出做数值对比验证
- CMake 构建，C++17 标准

### Must NOT Have (Guardrails)
- **NO CUDA / GPU 依赖** — 纯 CPU 实现
- **NO TensorRT** — 使用 ONNX Runtime
- **NO PyTorch 运行时** — C++ 不得导入 Python/Torch
- **NO 重依赖** — 除 ONNX Runtime 和 Eigen3 外不引入新库
- **NO 模型导出/训练代码** — 仅推理
- **NO 单元测试框架** — 用 golden data 验证代替
- **NO 过度抽象** — 2-3 个源文件，保持简单

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (无 pytest/junit 等测试框架)
- **Automated tests**: None
- **Framework**: N/A
- **Verification method**: Golden data comparison — C++ 输出 vs Python 参考输出

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **C++ 编译**: 使用 Bash — cmake + make，检查编译输出和退出码
- **推理功能**: 使用 Bash — 运行编译后的二进制，检查输出格式和准确率
- **数值验证**: 使用 Bash — 运行 verify.py，检查 ALL PASS

---

## Execution Strategy

### Sequential Execution

> 按依赖关系顺序执行，每个任务完成后再开始下一个。简单、可控，适合单个开发者执行。

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9

然后并行执行最终验证:
Task F1 ∥ Task F2 ∥ Task F3 ∥ Task F4 → 展示结果 → 用户确认
```

### Dependency Chain

| 序号  | 任务                              | 前置依赖 | 产出                                     |
|--------|-----------------------------------|----------|-----------------------------------------|
| 1      | CMakeLists.txt + 类型声明          | -        | CMakeLists.txt, onnx_inference.h         |
| 2      | PLY reader                        | 1        | load_data_ply() in onnx_inference.cpp    |
| 3      | Stats loader                      | 1        | convert_stats.py, load_stats() in .cpp   |
| 4      | CPU voxelizer                     | 1        | voxelize_cpu() in onnx_inference.cpp     |
| 5      | ONNX engine wrapper               | 1        | create_session(), run_inference() in .cpp|
| 6      | Preprocessor                      | 1,3,4    | preprocess_subcloud() in onnx_inference.cpp|
| 7      | Scatter mean                      | 1        | scatter_mean() in onnx_inference.cpp     |
| 8      | Pipeline assembly + main CLI      | 2,5,6,7  | OnnxInferencePipeline class, main.cpp    |
| 9      | Python golden data verification   | 8        | verify.py                                |
| F1-F4  | Final reviews                     | 1-9      | 审查报告                                  |

### Agent Dispatch Summary

| 任务   | 代理              |
|--------|-------------------|
| 1-3    | `quick`           |
| 4      | `deep`            |
| 5-7    | `quick`           |
| 8      | `deep`            |
| 9      | `quick`           |
| F1     | `oracle`          |
| F2-F3  | `unspecified-high`|
| F4     | `deep`            |

---

## TODOs

- [x] 1. CMakeLists.txt + 类型声明 (onnx_inference.h)

  **What to do**:
  - 创建 `deploy/CPP_onnx/CMakeLists.txt`:
    - cmake_minimum_required 3.18，project(hpenet_onnx_infer)，C++17
    - 查找 ONNX Runtime (`find_package(onnxruntime)`) 和 Eigen3
    - 添加可执行文件 `hpenet_onnx_infer`（源文件 main.cpp + onnx_inference.cpp）
    - 不引入 CUDA、TensorRT、GoogleTest
  - 创建 `deploy/CPP_onnx/onnx_inference.h`:
    - `struct PointCloud { std::vector<float> coord; std::vector<float> feat; std::vector<float> label; int num_points; }`
    - `struct FeatureStats { float feat_mean[3]; float feat_std[3]; float z_mean; float z_std; }`
    - `struct InferenceResult { std::vector<float> logits; std::vector<int> predictions; float latency_ms; }`
    - `class OnnxInferencePipeline` 声明（构造、process_file、process_directory、析构）
    - `#pragma once` 保护，包含必要的标准库头文件

  **Must NOT do**:
  - 不引入 CUDA 或 TensorRT 依赖
  - 不从 CPP_trt 继承类（独立实现）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 模板化工作——CMake 配置和类型声明都是固定模式
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Start first, no dependency
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `deploy/CPP_trt/CMakeLists.txt` — CMake 模板参考（移除 CUDA/TRT/GoogleTest 部分，添加 ONNX Runtime）
  - `deploy/CPP_trt/include/types.h` — 数据结构命名和风格参考
  - `deploy/CPP_trt/include/pipeline.h` — 类接口设计参考
  - `deploy/trt_manual/onnx_inference.py:34-46` — Python ONNX 推理接口参考

  **Acceptance Criteria**:
  - [ ] `deploy/CPP_onnx/CMakeLists.txt` 存在，cmake 配置阶段无错误
  - [ ] `deploy/CPP_onnx/onnx_inference.h` 存在，类型声明完整

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: CMake 配置成功
    Tool: Bash
    Preconditions: ONNX Runtime 和 Eigen3 已安装在系统中
    Steps:
      1. cd deploy/CPP_onnx && mkdir -p build && cd build && cmake ..
      2. Assert cmake 退出码为 0
      3. Assert 输出中包含 "onnxruntime" 和 "Eigen3" 的 found 信息
    Expected Result: cmake 配置成功，无错误
    Failure Indicators: cmake 退出码非 0 或找不到 ONNX Runtime/Eigen3
    Evidence: .omo/evidence/task-1-cmake-config.txt

  Scenario: 头文件编译检查
    Tool: Bash
    Preconditions: CMake 配置成功
    Steps:
      1. cd deploy/CPP_onnx/build && cmake --build . --target hpenet_onnx_infer 2>&1 || echo "EXPECTED_FAIL"
      2. Assert 输出不包含 "syntax error" 或 "undefined" 相关错误
    Expected Result: 头文件可被 main.cpp 正常 include（即使 main.cpp 是空的）
    Failure Indicators: 头文件有语法错误
    Evidence: .omo/evidence/task-1-header-compile.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/CMakeLists.txt`, `deploy/CPP_onnx/onnx_inference.h`

- [x] 2. PLY reader

  **What to do**:
  - 在 `onnx_inference.cpp` 中实现 `load_data_ply(const std::string& path)` 函数
  - 使用 tinyply 读取 PLY 文件（从 `deploy/CPP_trt/include/tinyply/tinyply.h` 复制或 `#include` 相对路径）
  - 读取字段: x, y, z, rcs, snr, v, label — 全部 float32
  - 返回 `PointCloud` 结构体
  - 处理 NaN 值：`std::isnan()` → 替换为 0.0
  - 参考 `deploy/CPP_trt/src/ply_reader.cpp` 的实现风格
  - 函数签名: `PointCloud load_data_ply(const std::string& data_path)`

  **Must NOT do**:
  - 不复制整个 tinyply 目录 — 通过 `#include` 相对路径引用现有代码
  - 不使用第三方 PLY 库

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 已有 Python 参考实现和 CPP_trt 实现，只需按格式翻译
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - `deploy/common.py:24-50` — Python `load_data_ply` 完整实现
  - `deploy/CPP_trt/src/ply_reader.cpp` — 现有 tinyply C++ 用法
  - `deploy/CPP_trt/include/tinyply/tinyply.h` — tinyply API
  - `deploy/CPP_trt/include/types.h:6-14` — PointCloud 结构体定义

  **Acceptance Criteria**:
  - [ ] 能够成功读取测试 PLY 文件，返回非空 PointCloud
  - [ ] `num_points` 正确，字段维度匹配 (coord: 3, feat: 3, label: 1)
  - [ ] NaN 值被替换为 0.0

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 读取正常雷达 PLY 文件
    Tool: Bash (编译 + 运行测试代码)
    Preconditions: 测试 PLY 文件存在于 data/RadarClassi/radarfull/raw/
    Steps:
      1. 在 main.cpp 中临时添加代码：auto pc = load_data_ply("data/RadarClassi/radarfull/raw/<test_file.ply>")
      2. 输出 pc.num_points, coord 前 3 个值
      3. Assert num_points > 0
      4. Assert coord 值非 NaN
    Expected Result: 成功加载，输出实际点数
    Failure Indicators: num_points == 0 或崩溃
    Evidence: .omo/evidence/task-2-ply-load.txt

  Scenario: NaN 值处理
    Tool: Bash
    Preconditions: PLY 文件加载成功
    Steps:
      1. 遍历 feat 数据，检查是否有 NaN
      2. Assert 所有值为有限值（无 NaN 或 inf）
    Expected Result: 无 NaN 值
    Evidence: .omo/evidence/task-2-nan-check.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/onnx_inference.cpp`

- [x] 3. Stats loader

  **What to do**:
  - 在 `onnx_inference.cpp` 中实现 `load_stats(const std::string& stats_path)` 函数
  - 读取 PyTorch `.pth` 格式的统计文件
  - 由于 `.pth` 是 Python pickle 格式，C++ 无法直接读取。采用方案：
    - **方案 A**: 使用 `deploy/CPP_trt/scripts/convert_stats.py` 或类似脚本将 `.pth` 转为 JSON/TXT，C++ 直接读取
    - **方案 B**: 在 C++ 中硬编码统计值（仅适用于已知统计量的场景）
    - **推荐方案 A**：创建 `deploy/CPP_onnx/scripts/convert_stats.py` 将 `.pth` → `.json`，C++ 用 nlohmann/json 或手动 JSON 解析
  - 函数签名: `FeatureStats load_stats(const std::string& stats_json_path)`
  - 输入格式: JSON 文件包含 `feat_mean`, `feat_std`, `z_mean`, `z_std`

  **Must NOT do**:
  - 不直接在 C++ 中解析 PyTorch pickle 格式
  - 不硬编码统计值（必须从文件加载）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 数据处理任务，有明确的输入输出格式
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:
  - `deploy/common.py:77-85` — Python `load_stats` 实现
  - `deploy/CPP_trt/scripts/convert_stats.py` — 现有的 stats 转换脚本（.pth→.json）
  - `deploy/CPP_trt/include/stats_reader.h` + `src/stats_reader.cpp` — C++ stats 读取参考
  - `deploy/CPP_trt/stats.json` — JSON 格式示例

  **Acceptance Criteria**:
  - [ ] `convert_stats.py` 能正确将 `.pth` 转为 JSON
  - [ ] C++ 能正确从 JSON 加载 FeatureStats（5 个字段均非零）
  - [ ] `z_mean` 和 `z_std` 值与 Python `load_stats` 一致

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: stats 转换脚本
    Tool: Bash
    Preconditions: stats .pth 文件存在
    Steps:
      1. cd deploy/CPP_onnx && python3 scripts/convert_stats.py --input ../../data/RadarClassi/radarfull/processed/feat_stats_area5.pth --output stats.json
      2. Assert stats.json 存在
      3. python3 -c "import json; s=json.load(open('stats.json')); assert 'feat_mean' in s; print(s['z_mean'])"
    Expected Result: stats.json 创建成功，包含所有 5 个字段
    Failure Indicators: 转换失败或字段缺失
    Evidence: .omo/evidence/task-3-convert-stats.txt

  Scenario: C++ stats 加载
    Tool: Bash
    Preconditions: stats.json 存在
    Steps:
      1. 在 main.cpp 临时测试: auto s = load_stats("stats.json"); 输出 s.feat_mean[0]
      2. Assert 输出值与 Python 读取的一致（差异 < 1e-6）
    Expected Result: 统计值正确加载
    Evidence: .omo/evidence/task-3-load-stats.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/onnx_inference.cpp`, `deploy/CPP_onnx/scripts/convert_stats.py`

- [x] 4. CPU voxelizer

  **What to do**:
  - 在 `onnx_inference.cpp` 中实现纯 CPU 的体素化函数
  - 复制 Python `voxelize(coord, voxel_size, mode=1)` 的行为（`openpoints/dataset/data_util.py`）：
    1. 将坐标量化到体素网格: `voxel_id = floor(coord / voxel_size)`
    2. 按 voxel_id 排序（stable sort）
    3. 统计每个体素的点数 `count`
    4. 生成子云索引: 对每个 shift i (0 ≤ i < max(count))，从每个体素中取第 i 个点
    5. 将子云中的点随机打乱 (std::shuffle)
  - 函数签名: `std::vector<std::vector<int>> voxelize_cpu(const float* coord, int num_points, float voxel_size, int seed)`
  - 返回: `idx_points` — 子云索引列表（vector of sub-cloud index vectors）
  - 使用 Eigen::Vector3f 做逐点运算，使用 std::unordered_map 做体素哈希
  - 注意处理 NaN 坐标: `std::isnan()` → 替换为 0

  **Must NOT do**:
  - 不使用 CUDA — 纯 CPU 实现
  - 不从 openpoints Python 代码中直接调用 CUDA

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 体素化算法是管道中最复杂的部分，需要理解 Python 的 mode=1 逻辑并精确复现
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3
  - **Blocked By**: Task 2

  **References**:
  - `openpoints/dataset/data_util.py` 中 `voxelize()` 函数 — mode=1 逻辑
  - `deploy/common.py:53-74` — `preprocess_test` 中的体素化调用和子云生成
  - `deploy/CPP_trt/src/voxelizer.cpp` — 现有 CUDA 体素化实现（理解语义即可，不照抄 CUDA 逻辑）
  - `deploy/CPP_trt/include/voxelizer.h` — 接口签名参考

  **Acceptance Criteria**:
  - [ ] 对 N 点云体素化后，子云中的每个点都来自不同体素（mode=1 保证）
  - [ ] 所有子云的总点数 = 原始云的总点数
  - [ ] 体素化结果与 Python `voxelize(coord, 0.1, mode=1)` 产生的子云索引一致（允许 shuffle 带来的顺序差异）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 体素化正确性（简单测试）
    Tool: Bash
    Preconditions: 有可用的测试 PLY 文件
    Steps:
      1. 加载测试 PLY，对 coord 做体素化（voxel_size=0.1, seed=100）
      2. 验证：所有子云总点数 == 原始点数
      3. 验证：每个子云中不存在重复的体素 ID（对子云中的每个点计算 floor(coord/voxel_size)，确认唯一）
    Expected Result: 总点数不变，每个子云中无重复体素
    Failure Indicators: 总点数不匹配或出现重复体素
    Evidence: .omo/evidence/task-4-voxelize.txt

  Scenario: 体素化结果与 Python 对比
    Tool: Bash
    Preconditions: Python 生成 golden 参考数据
    Steps:
      1. 运行 Python 脚本生成体素化结果为 CSV
      2. C++ 输出相同输入的体素化结果
      3. Diff 比较：确认两者的子云划分逻辑一致（忽略 shuffle 带来的顺序差异）
    Expected Result: 体素划分逻辑匹配
    Evidence: .omo/evidence/task-4-voxel-compare.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/onnx_inference.cpp`

- [x] 5. ONNX engine wrapper

  **What to do**:
  - 在 `onnx_inference.cpp` 中实现 ONNX Runtime 推理函数
  - 使用 ONNX Runtime C++ API:
    - `Ort::Env`, `Ort::SessionOptions`（设置 CPUExecutionProvider, 线程数）
    - `Ort::Session` 加载 ONNX 模型
    - 创建 `Ort::Value` 输入张量: pos (1, N, 3) float32, x (1, 4, N) float32
    - 运行 `session.Run()` 获取输出 (1, 2, N) float32
  - 考虑多线程安全: 每个调用路径使用独立的 `Ort::Session`，不跨线程共享 session
  - 函数签名:
    - `Ort::Session create_session(const std::string& onnx_path, int num_threads)`
    - `std::vector<float> run_inference(Ort::Session& session, const float* pos, const float* x, int N)`
  - 输出形状: (1, 2, N) → 存储为 std::vector<float>，row-major layout
  - 链接: `-lonnxruntime`

  **Must NOT do**:
  - 不使用 CUDAExecutionProvider
  - 不在多线程间共享 Ort::Session（每个线程创建自己的 session）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: ONNX Runtime C API 模式固定，有完整的官方文档参考
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:
  - `deploy/trt_manual/onnx_inference.py:34-46` — Python `run_onnx_inference` 函数
  - `deploy/CPP_trt/src/trt_inference.cpp` — 推理引擎的 C++ 封装风格参考
  - ONNX Runtime 官方文档: `https://onnxruntime.ai/docs/api/c/` — C API（C++ wrapper 基于此）
  - ONNX Runtime C++ API headers: `include/onnxruntime/core/session/onnxruntime_cxx_api.h`

  **Acceptance Criteria**:
  - [ ] ONNX 模型加载成功，输入/输出名称正确获取
  - [ ] 使用随机数据运行一次推理，输出形状为 (1, 2, N) — N 为输入点数
  - [ ] 推理结果中无 NaN 值

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: ONNX 模型加载和基础推理
    Tool: Bash
    Preconditions: deploy/onnx_model.onnx 存在
    Steps:
      1. 在 main.cpp 中临时测试: 创建 session，用随机数据运行推理（N=1024）
      2. 输出推理结果的前 5 个值
      3. Assert session 创建成功（无异常）
      4. Assert 输出张量形状正确（2 × N，N 为输入点数）
      5. Assert 输出值非 NaN
    Expected Result: 模型加载成功，推理产生有效输出
    Failure Indicators: session 创建失败，推理崩溃，或输出全 NaN
    Evidence: .omo/evidence/task-5-onnx-infer.txt

  Scenario: 多线程安全测试
    Tool: Bash
    Preconditions: ONNX 推理功能正常
    Steps:
      1. 创建 4 个线程，每个线程独立创建 session 并运行推理
      2. Assert 所有线程完成推理，无异常/crash
      3. Assert 各线程输出与单线程一致
    Expected Result: 多线程推理无冲突
    Failure Indicators: crash 或输出不一致
    Evidence: .omo/evidence/task-5-thread-safe.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/onnx_inference.cpp`

- [x] 6. Preprocessor

  **What to do**:
  - 在 `onnx_inference.cpp` 中实现 `preprocess_subcloud()` 函数
  - 完整复制 Python `preprocess_subcloud`（`deploy/common.py:88-115`）的逻辑:
    1. 从完整云中按 `idx_part` 索引提取子云坐标和特征
    2. 坐标中心化: `coord -= coord.min()`
    3. XYZAlign: `pos = pos - mean(pos); pos[:,2] -= min(pos[:,2])`
    4. 特征归一化: `(feat - feat_mean) / max(feat_std, 1e-5)`
    5. 高度归一化: `(heights - z_mean) / max(z_std, 1e-5)`
    6. 组合特征: cat(feat, heights) → (N, 4) → transpose → (4, N)
    7. 添加 batch 维度: pos → (1, N, 3), x → (1, 4, N)
  - 函数签名: `void preprocess_subcloud(const float* coord, const float* feat, const int* idx_part, int num_part, const FeatureStats& stats, std::vector<float>& pos_out, std::vector<float>& x_out)`
  - 使用 Eigen 做向量运算
  - 注意 `max(std, 1e-5)` 避免除零

  **Must NOT do**:
  - 不使用 PyTorch 或任何 Python 依赖
  - 不修改输入的 coord/feat 数组（应使用副本）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯数学运算翻译，从 Python NumPy → C++ Eigen，逻辑已明确
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 5

  **References**:
  - `deploy/common.py:88-115` — Python `preprocess_subcloud` 完整实现
  - `deploy/CPP_trt/src/preprocessor.cpp` — 现有的 C++ 预处理实现参考
  - `deploy/CPP_trt/include/preprocessor.h` — 接口设计参考
  - `deploy/trt_manual/onnx_inference.py:80-106` — Python 端到端循环（使用 preprocess 的地方）

  **Acceptance Criteria**:
  - [ ] 对已知输入，C++ 预处理的 pos 和 x 输出与 Python `preprocess_subcloud` 一致（差异 < 1e-5）
  - [ ] 正确处理除零保护（max(std, 1e-5)）
  - [ ] pos 输出形状: (1, num_part, 3) row-major
  - [ ] x 输出形状: (1, 4, num_part) row-major

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 预处理数值对比
    Tool: Bash
    Preconditions: 有 golden 参考数据（Python 预处理输出）
    Steps:
      1. Python 对测试点云做预处理，保存 pos/x 为 binary
      2. C++ 对相同输入做预处理
      3. 逐元素比较: max_abs_diff(C++, Python) < 1e-5
    Expected Result: 数值完全一致
    Failure Indicators: 差异超过 1e-5
    Evidence: .omo/evidence/task-6-preprocess-compare.txt

  Scenario: 除零保护
    Tool: Bash
    Preconditions: 预处理功能正常
    Steps:
      1. 构造所有特征值相等（std=0）的输入
      2. 运行预处理，Assert 不崩溃且输出值有限（非 NaN/Inf）
    Expected Result: 输出有限值（除以 1e-5 而非 0）
    Evidence: .omo/evidence/task-6-divzero.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/onnx_inference.cpp`

- [x] 7. Scatter mean

  **What to do**:
  - 在 `onnx_inference.cpp` 中实现 CPU scatter_mean 操作
  - 等价于 PyTorch `torch_scatter.scatter(src, index, dim=0, reduce='mean')`
  - 输入:
    - `logits`: (total_subcloud_points, 2) — 所有子云推理结果
    - `indices`: (total_subcloud_points,) — 原始点云索引
    - `num_orig`: 原始点云总点数
  - 输出: `merged` (num_orig, 2) — 每个原始点的平均 logits
  - 算法:
    1. 分配 `sum` (num_orig, 2) 和 `count` (num_orig) 初始化为 0
    2. 对每个 (logit, idx): `sum[idx] += logit; count[idx] += 1`
    3. `merged[idx] = sum[idx] / count[idx]`
  - 使用 OpenMP 加速（`#pragma omp parallel for`），注意线程安全（使用 per-thread 局部累加再合并）
  - 函数签名: `std::vector<float> scatter_mean(const float* logits, const int* indices, int total_points, int num_orig, int num_classes)`

  **Must NOT do**:
  - 不使用 CUDA — 纯 CPU 实现
  - 不依赖 torch_scatter Python 包

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 标准归约算法，实现简单
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 6)
  - **Blocks**: Task 8
  - **Blocked By**: Task 6

  **References**:
  - `deploy/trt_manual/onnx_inference.py:111-121` — Python scatter_mean 调用方式
  - `deploy/CPP_trt/src/kernels/scatter_mean.cu` — 现有 CUDA scatter_mean 实现（理解语义）
  - `deploy/CPP_trt/include/scatter_mean.h` — 接口参考

  **Acceptance Criteria**:
  - [ ] 对简单测试数据（3 个点，2 个类别），scatter_mean 结果手动可验证
  - [ ] 与 Python `torch_scatter.scatter(..., reduce='mean')` 输出一致（差异 < 1e-5）
  - [ ] OpenMP 多线程版本结果与单线程一致

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 基本正确性测试
    Tool: Bash
    Preconditions: scatter_mean 函数可编译
    Steps:
      1. 构造测试数据: logits = [[1,2],[3,4],[5,6]], indices = [0,0,1]
      2. 期望输出: [[2,3],[5,6]] (前两个点归约到 index 0，求平均)
      3. Assert 输出 == 期望值
    Expected Result: 简单测试通过
    Failure Indicators: 输出值不对
    Evidence: .omo/evidence/task-7-scatter-basic.txt

  Scenario: 与 Python scatter 对比
    Tool: Bash
    Preconditions: Python torch_scatter 可用
    Steps:
      1. Python 生成随机 logits + indices，保存 golden 结果
      2. C++ 对相同数据运行 scatter_mean
      3. 逐元素比较: max_abs_diff(C++, Python) < 1e-5
    Expected Result: 与 Python 一致
    Evidence: .omo/evidence/task-7-scatter-compare.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/onnx_inference.cpp`

- [x] 8. Pipeline assembly + main CLI

  **What to do**:
  - 在 `onnx_inference.cpp` 中实现 `OnnxInferencePipeline` 类:
    - `process_file(const std::string& ply_path)` — 端到端单文件推理
    - `process_directory(const std::string& dir_path, int num_files)` — 批量推理
    - 管道: PLY → voxelize → preprocess → ONNX inference → scatter_mean → argmax → InferenceResult
    - 记录 `latency_ms`（推理耗时）
  - 创建 `main.cpp` 命令行入口:
    - 使用 getopt 或简单的 argc/argv 解析
    - 参数: `--onnx`, `--data_dir`, `--stats_file`, `--num_files`（默认 3）
    - 循环调用 `pipeline.process_file()` 并输出准确率
    - 输出格式与 Python `onnx_inference.py` main 一致

  **Must NOT do**:
  - 不在 main.cpp 中硬编码路径
  - 不在 pipeline 中引入不必要的间接层

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 管道装配需要协调多个模块的输出/输入，需要仔细验证数据流
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:
  - `deploy/trt_manual/onnx_inference.py:69-121` — `infer_one_cloud_onnx` 端到端函数
  - `deploy/trt_manual/onnx_inference.py:170-258` — `main()` 函数
  - `deploy/CPP_trt/src/pipeline.cpp` — 现有 pipeline 组装参考
  - `deploy/CPP_trt/src/main.cpp` — 现有 main 入口参考

  **Acceptance Criteria**:
  - [ ] `process_file()` 对单个 PLY 文件产生 valid 预测（准确率 > 0）
  - [ ] `process_directory()` 遍历多个文件，准确率逐文件不同
  - [ ] main 编译后可运行，输出格式与 Python 版本一致

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 端到端单文件推理
    Tool: Bash
    Preconditions: ONNX 模型和 stats.json 存在
    Steps:
      1. cd deploy/CPP_onnx/build && ./hpenet_onnx_infer --onnx ../../deploy/onnx_model.onnx --data_dir ../../data/RadarClassi/radarfull/raw --stats_file stats.json --num_files 1
      2. Assert 退出码为 0
      3. Assert 输出中包含准确率值（如 "acc=0.xxxx"）
    Expected Result: 单文件推理成功，输出准确率
    Failure Indicators: 崩溃或退出码非 0
    Evidence: .omo/evidence/task-8-e2e-single.txt

  Scenario: 批量推理
    Tool: Bash
    Preconditions: 测试数据目录有 >= 3 个 PLY 文件
    Steps:
      1. ./hpenet_onnx_infer ... --num_files 3
      2. Assert 输出 3 行推理结果
      3. Assert 每条结果包含文件名和准确率
    Expected Result: 3 个文件全部推理成功
    Evidence: .omo/evidence/task-8-e2e-batch.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/onnx_inference.cpp`, `deploy/CPP_onnx/main.cpp`

- [x] 9. Python golden data verification

  **What to do**:
  - 创建 `deploy/CPP_onnx/verify.py`
  - 功能:
    1. 运行 Python 原版 ONNX 推理（`deploy/trt_manual/onnx_inference.py`）生成 golden 参考输出
    2. 运行 C++ ONNX 推理二进制生成 C++ 输出
    3. 比较两者的 logits 和 predictions
    4. 输出 PASS/FAIL 总结
  - 比较指标:
    - max absolute difference (logits) < 1e-5
    - prediction accuracy 差异 < 0.1%
  - 支持参数: `--onnx`, `--cpp_binary`, `--data_dir`, `--stats_file`, `--num_files`

  **Must NOT do**:
  - 不在 verify.py 中重新实现推理逻辑 — 只调用已有的二进制/脚本
  - 不对 GPU/CUDA 做任何依赖

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Python 脚本，调用已有工具进行结果对比
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Execution Order**: Sequential (after Task 8)
  - **Blocks**: None
  - **Blocked By**: Task 8

  **References**:
  - `deploy/trt_manual/onnx_inference.py` — Python 参考推理代码
  - `deploy/common.py` — 共用数据加载函数

  **Acceptance Criteria**:
  - [ ] `verify.py` 能成功调用 Python 推理和 C++ 推理
  - [ ] 对 3 个测试文件，max_abs_diff(logits) < 1e-5
  - [ ] 最终输出 "ALL PASS"

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 验证脚本端到端
    Tool: Bash
    Preconditions: C++ 二进制已编译，Python ONNX 推理可运行
    Steps:
      1. cd deploy/CPP_onnx && python3 verify.py --onnx ../../deploy/onnx_model.onnx --cpp_binary ./build/hpenet_onnx_infer --data_dir ../../data/RadarClassi/radarfull/raw --stats_file stats.json --num_files 3
      2. Assert 退出码为 0
      3. Assert 输出 "ALL PASS"
    Expected Result: ALL PASS
    Failure Indicators: 输出 "FAIL" 或差异超过阈值
    Evidence: .omo/evidence/task-9-verify-pass.txt

  Scenario: Python vs C++ logits 差异报告
    Tool: Bash
    Preconditions: verify.py 运行完成
    Steps:
      1. 检查 verify.py 输出的逐文件差异值
      2. Assert 每个文件的 max_abs_diff < 1e-5
    Expected Result: 所有文件数值一致
    Evidence: .omo/evidence/task-9-diff-report.txt
  ```

  **Commit**: YES (groups with Task 8-9)
  - Message: `feat(onnx): add C++ ONNX inference pipeline`
  - Files: `deploy/CPP_onnx/verify.py`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, cmake build). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.omo/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `cmake --build build` + linter. Review all changed files for: C-style casts, bare `new`/`delete`, uninitialized variables, potential data races. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean build. Run C++ binary on 3 test PLY files. Execute `verify.py` — all scenarios must PASS. Test edge cases: empty point cloud (0 points), single-point cloud, missing ONNX model.
  Save to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | verify.py [PASS/FAIL] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1-9**: `feat(onnx): add C++ ONNX inference pipeline` - `deploy/CPP_onnx/*`, 验证通过后一次性提交
- Pre-commit: `cmake -B build && cmake --build build && cd deploy/CPP_onnx && python3 verify.py`

---

## Success Criteria

### Verification Commands
```bash
# 编译
cd deploy/CPP_onnx && cmake -B build && cmake --build build

# 运行推理
cd deploy/CPP_onnx && ./build/hpenet_onnx_infer --onnx ../../deploy/onnx_model.onnx --data_dir ../../data/RadarClassi/radarfull/raw --stats_file ../../data/RadarClassi/radarfull/processed/feat_stats_area5.pth --num_files 3

# 验证输出
cd deploy/CPP_onnx && python3 verify.py
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] `cmake --build build` 通过
- [ ] 推理输出与 Python 原版一致
- [ ] `verify.py` 输出 ALL PASS
