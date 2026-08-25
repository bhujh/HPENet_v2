# TensorRT 8.6 ↔ 10.x 版本兼容 + Windows 跨平台适配

## TL;DR

> **Quick Summary**: 使 deploy/CPP/ 构建系统和源码同时适配 Linux（TRT 8.6/10.x）和 Windows（TRT 10.x MSVC）。C++ 源码已确认跨平台 source-compatible，仅需更新 CMake 构建文件 + 添加版本守卫 + Windows 路径/平台守卫。
> 
> **Deliverables**:
> - `cmake/FindTensorRT.cmake` — 添加 TRT 10.x 搜索路径（Linux + Windows）
> - `CMakeLists.txt` — 平台守卫（RPATH/MSVC）+ TRT 版本打印 + `CMAKE_CUDA_ARCHITECTURES` cache variable
> - `src/trt_engine.cpp` — `static_assert(NV_TENSORRT_MAJOR >= 8)` 版本守卫
> 
> **Estimated Effort**: Short
> **Parallel Execution**: YES — Wave 1 可并行修改 3 文件
> **Critical Path**: 文件修改（Wave 1）→ Linux 构建验证（Wave 2）+ Windows 构建验证（Wave 2）

---

## Context

### Original Request
1. 使 deploy/CPP/ 代码适配 TensorRT 8.6 和 TensorRT 10.x
2. 同时支持 Windows（MSVC + CUDA 12.x + TRT 10.x）构建编译

### 三机器部署拓扑

| 机器 | OS | CUDA | TRT | GPU | Arch |
|------|-----|------|-----|-----|------|
| 机器 A（当前开发机）| Linux | 11.8 | 8.6.1.6 | RTX 30xx/40xx | sm80/sm86/sm89 |
| 机器 B（TRT 10.x Linux）| Linux | 12.x | 10.9/10.16 | RTX 5090 | sm120 |
| 机器 C（TRT 10.x Windows）| Windows | 12.x | 10.16 | RTX 5060 | sm120 |

### Interview Summary
**Key Discussions (Linux)**:
- TRT 10.x 在另一台 CUDA 12.x 机器上运行，当前机器保持 CUDA 11.8 + TRT 8.6 不变
- TRT 10.x 机器 GPU 为 RTX 5090（sm120），需要 `-DCMAKE_CUDA_ARCHITECTURES="120"`
- 版本检测采用方案 A：`static_assert(NV_TENSORRT_MAJOR >= 8)` + 注释，不做 #ifdef 空壳
- .engine 文件在 TRT 10.x 机器上重新构建，不尝试兼容加载

**Key Discussions (Windows — 新增)**:
- Windows 环境: MSVC (Visual Studio 2022) + CUDA 12.x + TRT 10.16
- Windows GPU: RTX 5060（sm120）
- TRT 安装路径: `C:\TensorRT-10.16.1.11`（自定义路径，无空格）
- 构建方式: CMake 命令行（和 Linux 一致）
- 单元测试: 也需要在 Windows 上编译并运行 GoogleTest

**Research Findings**:
- 逐 API 对照 NVIDIA TRT 8→10 C++ Migration Guide 确认：全部 15 个 TRT API 调用在 TRT 8.6 和 10.x 间签名一致，无废弃/移除
- `nvinfer1::Dims::d[]` 类型从 `int32_t` 变为 `int64_t`（TRT 10.x），但代码仅做值赋值（`d[0]=1,d[1]=N`），`int→int64_t` 隐式转换正常工作
- `NV_TENSORRT_MAJOR` 宏通过 `#include <NvInfer.h>` → `NvInferVersion.h` 自动可用
- `libnvonnxparser.so`/`libnvinfer_plugin.so` 在 TRT 10.x 中仍作为共享库提供
- **Windows 代码可移植性**：C++ 源码无 Linux 特有系统调用（无 fork/exec/pthread），`std::filesystem`/`<chrono>`/等均为 C++17 标准库跨平台。CUDA kernel 和 Thrust 同样跨平台
- **Windows 构建差异**：CMake `find_library(nvinfer)` 自动处理 `.lib`/`.dll` 后缀；RPATH 为 Linux 特有（需 `if(UNIX)` 守卫）；MSVC 可能需要 `/MD` 动态 CRT 标志

### Metis Review
**Key Findings**:
- 确认 main.cpp 已使用真实推理流水线（非 stub）
- 建议明确 Scope IN/OUT 边界和测试策略
- 建议 CMAKE_CUDA_ARCHITECTURES 改为可覆盖变量以适应不同 GPU
- 建议将"10.9/10.16"统一表述为"10.x"

---

## Work Objectives

### Core Objective
使 deploy/CPP/ 构建系统跨平台支持 Linux（TRT 8.6/10.x）和 Windows（TRT 10.x MSVC），C++ 源码保持零修改。

### Concrete Deliverables
1. `cmake/FindTensorRT.cmake` — 搜索路径包含 TRT 10.x（Linux + Windows 双平台）
2. `CMakeLists.txt` — 平台守卫（RPATH UNIX-only / MSVC 标志）+ 版本打印 + arch cache variable
3. `src/trt_engine.cpp` — `static_assert(NV_TENSORRT_MAJOR >= 8)` 版本守卫

### Definition of Done
- [ ] Linux TRT 8.6: `cmake + make + ctest` 10/10 通过
- [ ] Linux TRT 10.x: 用户手动编译通过（另一台机器）
- [ ] Windows TRT 10.x: 用户手动编译 + ctest 通过（另一台机器）

### Must Have
- **所有平台**: TRT 8.6 Linux 环境回归零 breakage
- **所有平台**: CMake 阶段打印 TRT 检测版本号
- **所有平台**: `static_assert(NV_TENSORRT_MAJOR >= 8)` 版本守卫
- **Windows**: `find_library` 成功找到 nvinfer/nvinfer_plugin/nvonnxparser
- **Windows**: MSVC 编译 0 error, ctest 10/10 通过

### Must NOT Have (Guardrails)
- 不修改任何 `.cu` CUDA kernel 文件
- 不修改任何 Python 文件（deploy/*.py）
- 不添加 engine 版本兼容层
- 不考虑 CMake 现代化迁移（保留 `find_package(CUDA)`）
- 不做性能基准测试对比
- **不修改 C++ 源码逻辑**（仅添加 static_assert + 注释，无功能性变更）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed on Linux TRT 8.6.
> Windows 验证由用户在另一台机器上手动执行（参考下方 QA Scenarios）。

### Test Decision
- **Infrastructure exists**: YES (GoogleTest via FetchContent)
- **Automated tests**: Tests-after（编译验证 + 回归测试）
- **Framework**: GoogleTest (gtest)
- **Linux TRT 8.6**: Agent 自动构建 + ctest
- **Windows TRT 10.x**: 用户参照 Task 5/6 的 QA Scenarios 手动执行

### QA Policy
所有 Linux 验证步骤通过 Bash 命令执行，证据捕获到 `.omo/evidence/`。

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1（并行修改 3 文件 + 平台适配）:
├── Task 1: FindTensorRT.cmake — 添加 TRT 10.x 搜索路径（Linux + Windows）
├── Task 2: CMakeLists.txt — 平台守卫 + TRT 版本打印 + arch cache variable
├── Task 3: trt_engine.cpp — static_assert 版本守卫

Wave 2（Linux 验证 — 依赖 Wave 1）:
├── Task 4: Linux TRT 8.6: 干净构建 + ctest 回归验证

Wave 3（Windows 验证 — 用户手动，依赖 Wave 1）:
├── Task 5: Windows TRT 10.x: 构建编译验证（用户手动）
└── Task 6: Windows TRT 10.x: ctest 单元测试（用户手动）

Wave FINAL（并行审核）:
├── Task F1: 计划合规审计 (oracle)
├── Task F2: 代码质量审查 (unspecified-high)
├── Task F3: 功能验证 (unspecified-high)
└── Task F4: 范围忠实性检查 (deep)
```

### Dependency Matrix
- **1, 2, 3** → 4 → F1-F4
- **1, 2, 3** → 5, 6 (Windows 验证，用户手动)

### Agent Dispatch Summary
- **Wave 1**: 3 tasks → `quick`
- **Wave 2**: 1 task → `quick`（代理自动）
- **Wave 3**: 2 tasks → 用户手动（代理提供 QA Scenarios）
- **Final**: 4 tasks → oracle, unspecified-high, deep

---

## TODOs

- [x] 1. **`cmake/FindTensorRT.cmake` — 添加 TRT 10.x 搜索路径（Linux + Windows）**

  **What to do**:
  - 在现有 Linux `_TRT_SEARCH_PATHS` 列表中添加 TRT 10.x Linux 路径：
    ```cmake
    /usr/local/TensorRT-10.9.0.34
    /usr/local/TensorRT-10.16.0.34
    /usr/local/TensorRT-10.15.1.34
    ```
  - 新增 Windows 搜索路径分支（CMake 使用正斜杠）：
    ```cmake
    if(WIN32)
      list(APPEND _TRT_SEARCH_PATHS
        "C:/TensorRT-10.16.1.11"
        "C:/TensorRT-10.16.0.34"
        "C:/TensorRT-10.9.0.34"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
      )
    endif()
    ```
  - PATH_SUFFIXES 已包含 `lib`（Windows .lib 导入库路径），无需修改
  - 保持现有 TRT 8.x 路径不变（向后兼容）

  **Must NOT do**:
  - 不改变 REQUIRED_VARS 逻辑（`nvinfer` 仍是唯一必需库）
  - 不修改版本读取逻辑（已自动兼容 10.x）
  - 不在 PATH_SUFFIXES 中添加 `lib64`（不适用于 Windows）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 单文件修改，追加 Linux + Windows 路径分支

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `deploy/CPP/cmake/FindTensorRT.cmake` — 当前文件，`_TRT_SEARCH_PATHS` 列表在 else 分支（第 44-51 行）

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: TRT 10.x Linux 路径出现在搜索列表中
    Tool: Bash (grep)
    Preconditions: 文件已修改
    Steps:
      1. grep 'TensorRT-10\.' deploy/CPP/cmake/FindTensorRT.cmake
    Expected Result: 至少找到 3 个 TRT 10.x 版本路径
    Failure Indicators: grep 返回空
    Evidence: .omo/evidence/task-1-trt10-paths.txt

  Scenario: Windows 路径分支存在
    Tool: Bash (grep)
    Steps:
      1. grep 'C:/TensorRT' deploy/CPP/cmake/FindTensorRT.cmake
    Expected Result: 至少找到 1 个 Windows TRT 路径
    Evidence: .omo/evidence/task-1-windows-paths.txt

  Scenario: TRT 8.x 路径仍然保留
    Tool: Bash (grep)
    Steps:
      1. grep 'TensorRT-8\.' deploy/CPP/cmake/FindTensorRT.cmake
    Expected Result: 原有 8.x 路径全部保留
    Evidence: .omo/evidence/task-1-trt8-paths.txt
  ```

- [x] 2. **`CMakeLists.txt` — 平台守卫 + TRT 版本打印 + `CMAKE_CUDA_ARCHITECTURES` cache variable**

  **What to do**:
  - 在 `find_package(TensorRT REQUIRED)` 之后添加 TRT 版本打印：
    ```cmake
    message(STATUS "TensorRT version: ${TensorRT_VERSION}")
    ```
  - 将硬编码的 `set(CMAKE_CUDA_ARCHITECTURES "80;86;89")` 改为 cache variable：
    ```cmake
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING
        "CUDA architecture codes (e.g., 80;86;89 for Ampere/Ada, 120 for Blackwell)")
    ```
  - RPATH 添加 `if(UNIX)` 守卫（RPATH 在 Windows 无意义）：
    ```cmake
    if(UNIX)
      set_target_properties(hpenet_trt_infer PROPERTIES
        BUILD_RPATH "${TensorRT_ROOT}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64"
        INSTALL_RPATH "${TensorRT_ROOT}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64"
      )
    endif()
    ```
  - 可选：添加 Windows MSVC 运行时库标志说明（通常 CUDA Toolkit 自动处理）

  **Must NOT do**:
  - 不修改 `find_package(CUDA)` 或 `CMP0146` 策略
  - 不修改 `find_package(TensorRT REQUIRED)` 逻辑
  - 不添加 `target_compile_definitions`（NV_TENSORRT_MAJOR 已由 NvInfer.h 提供）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 单文件 CMake 修改，添加 if(UNIX) 守卫 + cache var

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `deploy/CPP/CMakeLists.txt` — CMAKE_CUDA_ARCHITECTURES 在第 13 行，RPATH 在第 58-61 行

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: CMake 配置阶段打印 TRT 版本 (Linux)
    Tool: Bash
    Preconditions: 干净 build 目录
    Steps:
      1. cd deploy/CPP/build && cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1 | grep -i "TensorRT version"
    Expected Result: 输出包含 "TensorRT version: 8.6.1.6"
    Failure Indicators: 无版本输出
    Evidence: .omo/evidence/task-2-version-print.txt

  Scenario: CMAKE_CUDA_ARCHITECTURES 可被覆盖
    Tool: Bash
    Steps:
      1. cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="90" 2>&1
      2. grep 'CMAKE_CUDA_ARCHITECTURES' CMakeCache.txt
    Expected Result: CMakeCache.txt 中 CUDA_ARCHITECTURES:STRING=90
    Evidence: .omo/evidence/task-2-arch-override.txt

  Scenario: RPATH 被 UNIX 守卫
    Tool: Bash (grep)
    Steps:
      1. grep -A2 'if(UNIX)' deploy/CPP/CMakeLists.txt
    Expected Result: RPATH 设置被 `if(UNIX)` 包裹
    Evidence: .omo/evidence/task-2-rpath-guard.txt
  ```

- [x] 3. **`src/trt_engine.cpp` — 添加 `static_assert` 版本守卫 + 兼容性注释**

  **What to do**:
  - 在文件开头的 `#include` 之后添加 TRT 版本守卫：
    ```cpp
    // TensorRT 版本兼容性守卫
    // NV_TENSORRT_MAJOR 由 <NvInfer.h> → <NvInferVersion.h> 提供
    // 支持版本: 8.x (name-based tensor API, enqueueV3)
    //         10.x (同一 API, Dims::d[] 变为 int64_t, source-compatible)
    static_assert(NV_TENSORRT_MAJOR >= 8,
        "HPENet V2 requires TensorRT 8.0 or later. "
        "Detected NV_TENSORRT_MAJOR=" STRINGIFY(NV_TENSORRT_MAJOR));
    // 注意: 需要一个 STRINGIFY 宏将数字转为字符串
    ```
  - 定义一个 `STRINGIFY` 辅助宏（如果没有）：
    ```cpp
    #define STRINGIFY_IMPL(x) #x
    #define STRINGIFY(x) STRINGIFY_IMPL(x)
    ```
  - 添加注释块记录 TRT 8.6 ↔ 10.x API 兼容性分析

  **Must NOT do**:
  - 不添加 `#if NV_TENSORRT_MAJOR >= 10` 条件编译（用户选择方案 A）
  - 不修改任何函数实现

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 单文件添加 static_assert + 注释

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:
  - `deploy/CPP/src/trt_engine.cpp` — 目标文件, #include 区域在第 1-8 行
  - `/usr/local/TensorRT-8.6.1.6/include/NvInferVersion.h` — NV_TENSORRT_MAJOR 宏定义

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: static_assert 编译通过（TRT 8.6）
    Tool: Bash
    Preconditions: Wave 1 所有修改完成
    Steps:
      1. cd deploy/CPP/build && cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1
      2. make -j$(nproc) 2>&1 | grep -i "error"
    Expected Result: 编译 0 error
    Failure Indicators: static_assert 触发（仅当 NV_TENSORRT_MAJOR < 8 时）
    Evidence: .omo/evidence/task-3-build-ok.txt

  Scenario: static_assert 注释记录了 API 兼容性
    Tool: Bash (grep)
    Steps:
      1. grep -c "NV_TENSORRT_MAJOR" deploy/CPP/src/trt_engine.cpp
    Expected Result: 至少 2 处引用（static_assert + 注释）
    Evidence: .omo/evidence/task-3-assert-found.txt
  ```

- [x] 4. **干净构建 + ctest 回归验证**

  **What to do**:
  - 执行完整构建验证流程：
    1. 删除旧的 build 目录
    2. 创建新 build 目录
    3. cmake 配置
    4. make 编译
    5. ctest 运行测试
  - 确认编译 0 error, 0 warning（已知警告除外）
  - 确认 GoogleTest 10/10 全部通过

  **Must NOT do**:
  - 不修改任何文件

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 验证任务，无需代码修改

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Tasks 1-3)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 1, 2, 3

  **References**:
  - `deploy/CPP/tests/qa_evidence.txt` — 上次 QA 验证记录

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: 干净构建
    Tool: Bash
    Preconditions: deploy/CPP/build/ 目录不存在
    Steps:
      1. mkdir -p deploy/CPP/build && cd deploy/CPP/build
      2. cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1
      3. make -j$(nproc) 2>&1
    Expected Result: 4/4 targets built, 0 errors
    Failure Indicators: 编译错误或链接错误
    Evidence: .omo/evidence/task-4-build-log.txt

  Scenario: ctest 全部通过
    Tool: Bash
    Steps:
      1. cd deploy/CPP/build && ctest --output-on-failure 2>&1
    Expected Result: 10/10 tests passed
    Failure Indicators: 任何测试失败
    Evidence: .omo/evidence/task-4-ctest.txt

  Scenario: TRT 版本打印
    Tool: Bash
    Steps:
      1. cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1 | grep -i "version"
    Expected Result: "TensorRT version: 8.6.1.6" 出现在输出中
    Evidence: .omo/evidence/task-4-version.txt
  ```

  **Commit**: YES
  - Message: `build(cmake): cross-platform cmake for TRT 8.x/10.x on Linux & Windows`
  - Files:
    - `deploy/CPP/cmake/FindTensorRT.cmake`
    - `deploy/CPP/CMakeLists.txt`
    - `deploy/CPP/src/trt_engine.cpp`
  - Pre-commit: (验证任务应该在提交前完成)

- [~] 5. **Windows TRT 10.x: 构建编译验证（用户手动）**

  **What to do**:
  > 此任务由用户在 Windows 机器 C 上手动执行。此处提供精确的 QA Scenarios 供用户参照。
  - 用户需安装: Visual Studio 2022 + CUDA 12.x + TensorRT 10.16（路径 `C:\TensorRT-10.16.1.11`）
  - 构建流程：
    ```
    # 在 PowerShell 或 cmd 中
    cd deploy\CPP
    rmdir /s /q build
    mkdir build && cd build
    cmake .. -G "Visual Studio 17 2022" -A x64 ^
      -DTENSORRT_ROOT=C:/TensorRT-10.16.1.11 ^
      -DCMAKE_CUDA_ARCHITECTURES="120"
    cmake --build . --config Release -j
    ```

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 用户手动执行，代理仅提供 QA Scenarios 指导

  **Parallelization**:
  - **Can Run In Parallel**: NO (用户手动，独立于代理工作）
  - **Parallel Group**: Wave 3 (after Tasks 1-3)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2, 3

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Windows CMake 配置成功
    Tool: cmd.exe (用户手动)
    Preconditions: Visual Studio 2022 + CUDA 12.x + TRT 10.16 已安装
    Steps:
      1. cd deploy\CPP\build
      2. cmake .. -G "Visual Studio 17 2022" -A x64 -DTENSORRT_ROOT=C:/TensorRT-10.16.1.11 -DCMAKE_CUDA_ARCHITECTURES="120"
    Expected Result: "Configuring done" + "Generating done"，无 FATAL_ERROR
    Failure Indicators: "Could NOT find TensorRT" 或 "CUDA compiler not found"
    Evidence: 用户截图或复制 cmake 输出

  Scenario: Windows MSVC 编译成功
    Tool: cmd.exe (用户手动)
    Steps:
      1. cmake --build . --config Release -j
    Expected Result: 4/4 targets built, 0 errors
    Failure Indicators: 编译错误或链接错误（特别是 nvinfer.lib 找不到）
    Evidence: 用户截图或复制 build 输出
  ```

- [~] 6. **Windows TRT 10.x: ctest 单元测试（用户手动）**

  **What to do**:
  > 此任务由用户在 Windows 机器 C 上手动执行。
  - 在 Task 5 构建成功后，运行 ctest：
    ```
    cd deploy\CPP\build
    ctest --output-on-failure -C Release
    ```

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 用户手动执行，代理仅提供 QA Scenarios 指导

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 5, same user session)
  - **Parallel Group**: Wave 3 (after Tasks 1-3)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2, 3

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Windows ctest 全部通过
    Tool: cmd.exe (用户手动)
    Steps:
      1. cd deploy\CPP\build && ctest --output-on-failure -C Release
    Expected Result: 10/10 tests passed (如果 GoogleTest FetchContent 在 Windows 上成功下载)
    Failure Indicators: 测试失败或 FetchContent 下载超时（GFW 问题）
    Evidence: 用户截图或复制 ctest 输出
  ```

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan. Verify each "Must Have" is implemented. Check each "Must NOT Have" for violations.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Review changed files: check for proper format, comment style, no unused imports, no hardcoded paths.
  Output: `Files [N clean/N issues] | VERDICT`

- [x] F3. **Functional Verification** — `unspecified-high`
  Execute build + ctest from scratch, confirm 0 error + 10/10 tests.
  Output: `Build [PASS/FAIL] | Tests [N/N] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  Verify only 3 files changed (still 3, even with cross-platform expansion — both Linux/Windows changes in same files). No .cu files, no Python files, no engine compatibility layer added.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **1**: `build(cmake): cross-platform cmake for TRT 8.x/10.x on Linux & Windows`
  - `deploy/CPP/cmake/FindTensorRT.cmake`
  - `deploy/CPP/CMakeLists.txt`
  - `deploy/CPP/src/trt_engine.cpp`

---

## Success Criteria

### Linux TRT 8.6 自动验证
```bash
cd deploy/CPP && rm -rf build && mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1 | grep -i "version"
make -j$(nproc)
ctest --output-on-failure
```

### Windows TRT 10.x 用户手动验证（参照执行）
```cmd
cd deploy\CPP
rmdir /s /q build
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DTENSORRT_ROOT=C:/TensorRT-10.16.1.11 -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j
ctest --output-on-failure -C Release
```

### Final Checklist
- [ ] Linux TRT 8.6: 编译 0 error, GoogleTest 10/10 通过
- [ ] Linux TRT 8.6: TRT 版本号在 CMake 输出中可见
- [ ] Linux TRT 8.6: `static_assert(NV_TENSORRT_MAJOR >= 8)` 存在于 `trt_engine.cpp` 中
- [ ] Windows TRT 10.x: 编译 0 error（用户手动验证）
- [ ] Windows TRT 10.x: ctest 10/10 通过（用户手动验证）
- [ ] 只有 3 个文件被修改（FindTensorRT.cmake + CMakeLists.txt + trt_engine.cpp）
- [ ] 不涉及 .cu kernel / Python / engine 兼容层
