# C++ 部署代码 Windows/VS2022/CUDA12.8/TRT10 兼容性修复

## TL;DR

> **Quick Summary**: 修复 `deploy/CPP/` 中 7 个文件，使其在 Windows + VS 2022 17.8+ + CUDA 12.8 + TensorRT 10 环境下成功编译链接。
>
> **Deliverables**:
> - TensorRT 10 UniquePtr API 适配（trt_engine.h/cpp, trt_inference.h/cpp）
> - CMake 构建系统现代化（CUDAToolkit、MSVC CRT、DLL 部署）
> - FindTensorRT 模块 TRT 10 DLL 命名适配
> - 测试构建修复
>
> **Estimated Effort**: Short（~50 行修改，7 个文件）
> **Parallel Execution**: NO — 用户要求顺序执行，避免阻塞干扰
> **Critical Path**: 全部顺序 → 无并行路径

---

## Context

### Original Request
用户要求按方案 A（升级至 VS 2022 + TRT 10 修复）生成详细修复工作计划，顺序执行。

### Interview Summary
**Key Discussions**:
- 6-agent 并行审查发现了两个致命冲突：CUDA 12.8 不支持 MSVC 2019（需升级至 VS 2022），TRT 10 的 `createInferRuntime()`/`deserializeCudaEngine()` 返回 `UniquePtr<T>` 而非裸指针
- 源代码层面极干净：0 个 POSIX API 问题，已正确使用 TRT 10 name-based API
- 构建系统需修复 4 个问题：废弃的 `find_package(CUDA)`、MSVC CRT 不匹配、缺失 Windows DLL 部署、测试硬编码架构

**Metis Review 补充发现**:
- `trt_inference.h/cpp` 中的 `IExecutionContext* context_` 同样需要修改（原计划遗漏）
- 采用策略 B（成员存储 `nvinfer1::UniquePtr<T>`）确保 RAII 安全
- TRT 10 在 Windows 上将 DLL 重命名为 `nvinfer_10.dll`，`FindTensorRT.cmake` 需适配
- Scope 严格限定：不修改 CUDA kernel、推理逻辑、`main.cpp` 已知 BUG

### Research Findings
- CUDA 12.5+ 要求 VS 2022 17.8+（NVIDIA 官方）
- TRT 10 UniquePtr<T> 使用自定义 deleter（`destroy()` 方法），不可用 `std::unique_ptr` 替代
- TRT 10 Windows DLL: `nvinfer_10.dll`, `nvinfer_plugin_10.dll`, `nvonnxparser_10.dll`
- TRT 10 `createExecutionContext()` 也返回 `UniquePtr<IExecutionContext>`
- MSVC CRT: TensorRT 预编译库使用 `/MD`（动态 CRT），项目必须匹配

---

## Work Objectives

### Core Objective
修改 `deploy/CPP/` 中 7 个文件约 50 行代码，消除所有 BLOCKER 级别的兼容性问题，使项目在 Windows + VS 2022 + CUDA 12.8 + TensorRT 10 下成功编译。

### Concrete Deliverables
- `deploy/CPP/include/trt_engine.h` — 成员类型迁移
- `deploy/CPP/src/trt_engine.cpp` — 构造/析构/工厂方法适配
- `deploy/CPP/include/trt_inference.h` — 成员类型迁移
- `deploy/CPP/src/trt_inference.cpp` — 构造/析构适配
- `deploy/CPP/CMakeLists.txt` — CUDAToolkit + MSVC + DLL 部署
- `deploy/CPP/tests/CMakeLists.txt` — 架构 + CUDAToolkit 变量迁移
- `deploy/CPP/cmake/FindTensorRT.cmake` — TRT 10 DLL 命名 + Windows 路径优先

### Definition of Done
- [ ] Windows: `cmake -B build -G "Visual Studio 17 2022" -A x64` 配置成功（无 CMP0146/FindCUDA 错误）
- [ ] Windows: `cmake --build build --config Release` 编译成功（0 errors, 0 warnings）
- [ ] Linux 回归: `cmake -B build_linux && make -j$(nproc)` 编译成功（向后兼容）

### Must Have
- 所有 TRT 对象使用 `nvinfer1::UniquePtr<T>` 管理生命周期（RAII）
- CMake 使用现代 `find_package(CUDAToolkit)` 替代废弃的 `find_package(CUDA)`
- Windows 编译时 MSVC CRT 设置为 `MultiThreadedDLL`（匹配 TRT 预编译库）
- `FindTensorRT.cmake` 能搜索到 `nvinfer_10.lib`（Windows TRT 10 的 import library）
- Linux 回归编译不破坏

### Must NOT Have (Guardrails)
- **禁止修改** CUDA kernel 文件（`voxelize.cu`, `fnv_hash.cu`, `scatter_mean.cu`）
- **禁止修改** 推理逻辑（`pipeline.cpp`, `preprocessor.cpp`, `voxelizer.cpp`, `subcloud_utils.cpp` 等）
- **禁止修改** `main.cpp`（已知假数据 BUG，不在本次范围）
- **禁止修改** 第三方库 `tinyply.h`
- **禁止添加** `#if NV_TENSORRT_MAJOR >= 10` 的 TRT 8 兼容分支（仅支持 TRT 10）
- **禁止创建** Windows `.bat`/`.ps1` 构建脚本（除非 CMake 自身不足以构建）
- **禁止修改** 主项目的 `install.sh` 或 `openpoints/cpp/` CUDA extension 构建

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES（GoogleTest in `deploy/CPP/tests/`）
- **Automated tests**: Tests-after（修复后运行现有 `ctest` 验证）
- **Framework**: GoogleTest + CUDA
- **Agent-Executed QA**: ALWAYS — 每个任务编译后验证

### QA Policy
- **编译验证**: 每个任务完成后 `cmake --build` 验证无编译错误
- **Windows 环境**: 若无 Windows 环境，使用 Linux 交叉验证 TRT 10 兼容性（通过 `static_assert` + 语法检查）
- **Linux 回归**: 所有修改完成后在 Linux 上完整构建 + `ctest`

---

## Execution Strategy

### Sequential Execution（用户要求顺序执行）

```
Task 1 (trt_engine.h) ──→ Task 2 (trt_engine.cpp) ──→ Task 3 (trt_inference.h) ──→ 
Task 4 (trt_inference.cpp) ──→ Task 5 (CMakeLists.txt) ──→ Task 6 (tests/CMakeLists.txt) ──→ 
Task 7 (FindTensorRT.cmake) ──→ Verification
```

### Dependency Matrix

| Task | 依赖 | 原因 |
|------|------|------|
| 1. trt_engine.h | 无 | 类型定义先行 |
| 2. trt_engine.cpp | 1 | 使用 Task 1 定义的新类型 |
| 3. trt_inference.h | 1, 2 | 引用 `TrEngine::create_context()` 返回类型 |
| 4. trt_inference.cpp | 3 | 使用 Task 3 定义的新类型 |
| 5. CMakeLists.txt | 1-4 | 代码修改完成后调整构建系统 |
| 6. tests/CMakeLists.txt | 5 | 使用 Task 5 中的 `CUDAToolkit` 变量 |
| 7. FindTensorRT.cmake | 5 | 构建系统完整后调优模块搜索 |

---

## TODOs

- [x] 1. `trt_engine.h` — 成员变量类型迁移至 UniquePtr

  **What to do**:
  - 将第 57 行 `nvinfer1::IRuntime* runtime_ = nullptr;` 改为 `nvinfer1::UniquePtr<nvinfer1::IRuntime> runtime_;`
  - 将第 58 行 `nvinfer1::ICudaEngine* engine_ = nullptr;` 改为 `nvinfer1::UniquePtr<nvinfer1::ICudaEngine> engine_;`
  - 确认 `#include <NvInfer.h>` 已存在（提供 `UniquePtr` 定义）
  - 检查 `get()` / `get_engine()` 等访问器方法 — 裸指针返回方式不变（返回 `.get()`），但调用方不再负责 delete
  - 检查 `create_context()` 声明（返回 `IExecutionContext*`）— 暂不改返回类型，下个任务处理

  **Must NOT do**:
  - 不要改用 `std::unique_ptr`（TRT 的 DestroyDeleter 不兼容）
  - 不要添加 `#if NV_TENSORRT_MAJOR >= 10` 条件编译
  - 不要修改任何非成员类型的代码

  **Recommended Agent Profile**:
  - **Category**: `quick` — 简单类型替换，2 行修改
  - **Reason**: 纯头文件类型变更，无逻辑修改
  - **Skills**: 无特殊需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Task 1 (独立执行)
  - **Blocks**: Task 2（依赖 Task 1 的类型定义）
  - **Blocked By**: None

  **References**:
  - `deploy/CPP/include/trt_engine.h:6` — `#include <NvInfer.h>` 提供 `nvinfer1::UniquePtr<T>` 模板
  - `deploy/CPP/include/trt_engine.h:57-58` — 当前裸指针声明
  - `deploy/CPP/include/trt_engine.h:39-40` — `get()` 访问器（返回 `engine_.get()`），确认无需修改

  **Acceptance Criteria**:
  - [ ] `runtime_` 类型为 `nvinfer1::UniquePtr<nvinfer1::IRuntime>`
  - [ ] `engine_` 类型为 `nvinfer1::UniquePtr<nvinfer1::ICudaEngine>`
  - [ ] 头文件包含 `<NvInfer.h>`（已存在，验证即可）

  **QA Scenarios**:

  ```
  Scenario: 头文件语法检查（Happy path）
    Tool: Bash
    Preconditions: CUDA 12.8 + TRT 10 头文件已安装，或使用编译器语法检查
    Steps:
      1. echo '#include "deploy/CPP/include/trt_engine.h"' | g++ -std=c++17 -fsyntax-only -I deploy/CPP/include -I /path/to/TensorRT/include -I /path/to/CUDA/include -x c++ -
      2. 检查编译器输出
    Expected Result: 无语法错误，UniquePtr 类型被正确解析
    Failure Indicators: "'UniquePtr' is not a member of 'nvinfer1'" 或 "expected ';' after class"
    Evidence: .omo/evidence/task-1-header-syntax.txt
  ```

  **Commit**: NO（合入最终 commit）
- [x] 2. `trt_engine.cpp` — 构造/析构/工厂适配 UniquePtr

  **What to do**:
  - 第 56 行：`runtime_ = nvinfer1::createInferRuntime(logger);` → 保持不变（UniquePtr 从工厂函数接受赋值）
  - 第 62 行：`engine_ = runtime_->deserializeCudaEngine(serialized.data(), serialized.size());` → 保持不变
  - 第 86-94 行析构函数 `TrEngine::~TrEngine()`：**删除** `delete engine_;` 和 `delete runtime_;` 两行
  - 第 121 行 `create_context()`：`return engine_->createExecutionContext();` → TRT 10 中此方法返回 `UniquePtr<IExecutionContext>`，调用 `.release()` 返回裸指针：`return engine_->createExecutionContext().release();`
  - 更新第 22-24 行的注释：补充说明 TRT 10 `UniquePtr` 生命周期管理策略
  - 检查所有 `engine_->` 调用点 — 裸指针成员变为 UniquePtr 后，成员访问语法不变（`->` 仍可用）

  **Must NOT do**:
  - 不要在析构函数中用 `engine_.reset()` 或 `runtime_.reset()` — UniquePtr 析构自动调用
  - 不要修改 `create_context()` 的返回类型（头文件声明为 `IExecutionContext*`）
  - 不要删除第 22-27 行的 TRT 版本兼容性注释（只更新不删除）

  **Recommended Agent Profile**:
  - **Category**: `quick` — 4 处修改，纯机械替换
  - **Reason**: 删除 delete、添加 .release()，无逻辑判断
  - **Skills**: 无特殊需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Task 2 (独立执行)
  - **Blocks**: Task 3（TrEngine::create_context() 的返回语义影响 TrInference）
  - **Blocked By**: Task 1（成员类型定义）

  **References**:
  - `deploy/CPP/src/trt_engine.cpp:56` — `createInferRuntime` 调用
  - `deploy/CPP/src/trt_engine.cpp:62` — `deserializeCudaEngine` 调用
  - `deploy/CPP/src/trt_engine.cpp:86-94` — 析构函数，需删除 `delete engine_` 和 `delete runtime_`
  - `deploy/CPP/src/trt_engine.cpp:121` — `create_context()`，`.release()` 插入点
  - `deploy/CPP/src/trt_engine.cpp:22-27` — 版本兼容注释块

  **Acceptance Criteria**:
  - [ ] `TrEngine::~TrEngine()` 中无 `delete engine_` 和 `delete runtime_`
  - [ ] `create_context()` 返回 `engine_->createExecutionContext().release()`
  - [ ] 第 22-27 行注释已更新，提及 UniquePtr 策略

  **QA Scenarios**:

  ```
  Scenario: .release() 调用编译检查（Happy path）
    Tool: Bash
    Preconditions: trt_engine.h 已修改（Task 1）
    Steps:
      1. 使用 g++ -std=c++17 -fsyntax-only 编译 trt_engine.cpp
      2. 检查第 121 行 .release() 是否被正确识别
    Expected Result: 编译通过，无 "no member named 'release'" 错误
    Failure Indicators: "UniquePtr has no member release" 或类型不匹配
    Evidence: .omo/evidence/task-2-compile-check.txt

  Scenario: 析构函数无内存泄漏（逻辑检查）
    Tool: Bash (grep)
    Preconditions: trt_engine.cpp 已修改
    Steps:
      1. grep -n "delete" deploy/CPP/src/trt_engine.cpp
    Expected Result: 无 delete engine_ / delete runtime_（唯一 delete 在注释外）
    Failure Indicators: 仍有 delete engine_ 或 delete runtime_
    Evidence: .omo/evidence/task-2-no-delete.txt
  ```

  **Commit**: NO（合入最终 commit）
- [x] 3. `trt_inference.h` — context_ 成员迁移至 UniquePtr

  **What to do**:
  - 将第 75 行 `nvinfer1::IExecutionContext* context_ = nullptr;` 改为 `nvinfer1::UniquePtr<nvinfer1::IExecutionContext> context_;`
  - 确认 `#include <NvInfer.h>` 已存在（第 5 行）
  - 检查 `set_context()` 或 `get_context()` 等方法是否需要调整（当前代码似乎无此类方法）

  **Must NOT do**:
  - 不要改用 `std::unique_ptr`
  - 不要添加 `#if NV_TENSORRT_MAJOR >= 10` 条件编译

  **Recommended Agent Profile**:
  - **Category**: `quick` — 单行类型替换
  - **Reason**: 纯头文件类型变更，与 Task 1 同性质
  - **Skills**: 无特殊需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Task 3 (独立执行)
  - **Blocks**: Task 4（依赖 Task 3 的类型定义）
  - **Blocked By**: Task 2（TrEngine::create_context() 现在返回 .release() 的裸指针）

  **References**:
  - `deploy/CPP/include/trt_inference.h:5` — `#include <NvInfer.h>`
  - `deploy/CPP/include/trt_inference.h:75` — 当前裸指针声明
  - `deploy/CPP/include/trt_engine.h:38-40` — `create_context()` 声明（返回 `IExecutionContext*`）

  **Acceptance Criteria**:
  - [ ] `context_` 类型为 `nvinfer1::UniquePtr<nvinfer1::IExecutionContext>`
  - [ ] 头文件语法正确（通过 `-fsyntax-only`）

  **QA Scenarios**:

  ```
  Scenario: 头文件语法检查
    Tool: Bash
    Preconditions: trt_engine.h, trt_inference.h 均已修改
    Steps:
      1. echo '#include "deploy/CPP/include/trt_inference.h"' | g++ -std=c++17 -fsyntax-only -I deploy/CPP/include -I /path/to/TensorRT/include -I /path/to/CUDA/include -x c++ -
    Expected Result: 无语法错误
    Failure Indicators: 类型未定义错误
    Evidence: .omo/evidence/task-3-header-syntax.txt
  ```

  **Commit**: NO（合入最终 commit）
- [x] 4. `trt_inference.cpp` — 构造/析构适配 UniquePtr

  **What to do**:
  - 第 16 行构造函数：`context_ = engine_->create_context();` → TRT 10 中 `createExecutionContext()` 返回 `UniquePtr<IExecutionContext>`，但 `TrEngine::create_context()` 已改为返回 `.release()` 裸指针（Task 2）。**因此此处不需要 `.release()`**，但必须确认赋值兼容：
    - 若 `TrEngine::create_context()` 返回裸指针 → 需要 `context_.reset(engine_->create_context())` 来接管所有权
    - 实际上 `UniquePtr<IExecutionContext>` 无法直接从裸指针赋值，必须用 `.reset()`
  - 第 35 行析构函数：**删除** `delete context_;`（UniquePtr 自动析构）
  - 检查所有 `context_->` 调用点 — 成员访问语法不变（`->` 仍可用）

  **Must NOT do**:
  - 不要添加手动 `delete context_`
  - 不要修改 `TrInference::run()` 等推理方法

  **Recommended Agent Profile**:
  - **Category**: `quick` — 2 处修改
  - **Reason**: 赋值改为 `.reset()`，删除析构中的 delete
  - **Skills**: 无特殊需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Task 4 (独立执行)
  - **Blocks**: Task 5（代码修改完成，开始构建系统调整）
  - **Blocked By**: Task 3（成员类型定义）

  **References**:
  - `deploy/CPP/src/trt_inference.cpp:16` — 构造函数中 context_ 赋值
  - `deploy/CPP/src/trt_inference.cpp:35` — 析构函数中 `delete context_`
  - `deploy/CPP/include/trt_engine.h:38-40` — `TrEngine::create_context()` 声明

  **Acceptance Criteria**:
  - [ ] 构造函数中使用 `context_.reset(engine_->create_context())`
  - [ ] 析构函数中无 `delete context_`
  - [ ] 其余 `context_->` 调用点无变化

  **QA Scenarios**:

  ```
  Scenario: .reset() 调用编译检查（Happy path）
    Tool: Bash
    Steps:
      1. g++ -std=c++17 -fsyntax-only deploy/CPP/src/trt_inference.cpp -I deploy/CPP/include -I /path/to/TensorRT/include -I /path/to/CUDA/include
    Expected Result: 编译通过
    Failure Indicators: "no matching function for call to 'reset'"
    Evidence: .omo/evidence/task-4-compile-check.txt

  Scenario: 析构函数无手动 delete（逻辑检查）
    Tool: Bash (grep)
    Steps:
      1. grep -n "delete" deploy/CPP/src/trt_inference.cpp
    Expected Result: 无 delete context_
    Evidence: .omo/evidence/task-4-no-delete.txt
  ```

  **Commit**: NO（合入最终 commit）

- [x] 5. `CMakeLists.txt` — 构建系统现代化（CUDAToolkit + MSVC CRT + DLL 部署）

  **What to do**:
  **5a. 迁移 find_package(CUDA) → CUDAToolkit**:
  - 第 10 行：**删除** `cmake_policy(SET CMP0146 OLD)`（FindCUDA 模块已在 CMake 3.30+ 移除，策略无效）
  - 第 20 行：`find_package(CUDA REQUIRED)` → `find_package(CUDAToolkit REQUIRED)`
  - 第 60 行：`target_include_directories(... ${CUDA_INCLUDE_DIRS})` → `${CUDAToolkit_INCLUDE_DIRS}`
  - 第 61 行：`target_link_libraries(... ${CUDA_LIBRARIES})` → `CUDA::cudart`（现代 CMake target）

  **5b. 添加 MSVC CRT 匹配**:
  - 在 `project()` 之后（约第 5-6 行）插入：
    ```cmake
    if(MSVC)
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
    ```
  - TRT 预编译 `.lib` 使用动态 CRT（`/MD`），项目必须匹配

  **5c. 添加 Windows DLL 部署**:
  - 在第 68 行 `endif()` 之后，添加 `elseif(WIN32)` 分支：
    ```cmake
    elseif(WIN32)
      # 复制 TRT DLL 到输出目录（TRT 10: nvinfer_10.dll 等）
      file(GLOB TRT_DLLS "${TensorRT_ROOT}/lib/*.dll")
      foreach(DLL ${TRT_DLLS})
        configure_file("${DLL}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/" COPY_ONLY)
      endforeach()
      message(STATUS "Copied TensorRT DLLs to output directory")
    endif()
    ```

  **5d. 更新文档注释**:
  - 第 16 行 CMAKE_CUDA_ARCHITECTURES 注释：补充 sm_90 (Hopper) 和 sm_120 (Blackwell) 说明

  **Must NOT do**:
  - 不要删除 `if(UNIX)` 的 RPATH 配置
  - 不要修改 `CMAKE_CUDA_ARCHITECTURES` 的值（仅补充注释）
  - 不要修改变量缓存类型（`CACHE STRING`）

  **Recommended Agent Profile**:
  - **Category**: `quick` — CMake 语法修改，约 12 行
  - **Reason**: 纯配置文件修改，无编译逻辑
  - **Skills**: 无特殊需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Task 5 (独立执行)
  - **Blocks**: Task 6（tests/CMakeLists.txt 依赖 CUDAToolkit 变量）
  - **Blocked By**: Task 4（代码修改完成后进行构建系统调整）

  **References**:
  - `deploy/CPP/CMakeLists.txt:10` — 待删除的 `cmake_policy(SET CMP0146 OLD)`
  - `deploy/CPP/CMakeLists.txt:20` — 待替换的 `find_package(CUDA)`
  - `deploy/CPP/CMakeLists.txt:60-61` — 待替换的 `${CUDA_INCLUDE_DIRS}` / `${CUDA_LIBRARIES}`
  - `deploy/CPP/CMakeLists.txt:64-68` — `if(UNIX)` RPATH 块，在其后添加 `elseif(WIN32)`
  - `deploy/CPP/CMakeLists.txt:16-17` — CMAKE_CUDA_ARCHITECTURES 注释

  **Acceptance Criteria**:
  - [ ] `cmake --help-policy CMP0146` 无相关配置
  - [ ] 使用 `find_package(CUDAToolkit REQUIRED)`（非 `find_package(CUDA)`）
  - [ ] 使用 `if(MSVC)` 设置 `CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"`
  - [ ] RPATH 块 `if(UNIX)` 不变，新增 `elseif(WIN32)` DLL 复制逻辑

  **QA Scenarios**:

  ```
  Scenario: CMake 配置语法验证（Happy path — Linux 环境）
    Tool: Bash
    Preconditions: CUDA toolkit 已安装
    Steps:
      1. cmake -B /tmp/test-build -S deploy/CPP -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1
      2. 检查输出无 CMP0146 警告
      3. grep "CUDAToolkit" /tmp/test-build/CMakeCache.txt | head -3
    Expected Result: cmake 配置成功，无 "FindCUDA.cmake" 错误，CUDAToolkit 变量正确设置
    Failure Indicators: "Could not find FindCUDA.cmake" 或 "CUDAToolkit not found"
    Evidence: .omo/evidence/task-5-cmake-config.txt

  Scenario: MSVC CRT 设置存在性检查
    Tool: Bash (grep)
    Steps:
      1. grep "CMAKE_MSVC_RUNTIME_LIBRARY" deploy/CPP/CMakeLists.txt
    Expected Result: 输出 "MultiThreadedDLL"
    Failure Indicators: grep 无输出
    Evidence: .omo/evidence/task-5-msvc-crt.txt
  ```

  **Commit**: NO（合入最终 commit）
- [x] 6. `tests/CMakeLists.txt` — CUDA 架构 + CUDAToolkit 变量迁移

  **What to do**:
  **6a. 修复硬编码 CUDA_ARCHITECTURES**:
  - 第 19 行：`set_target_properties(${TEST_NAME} PROPERTIES CUDA_ARCHITECTURES "86")` → 改为 `set_target_properties(${TEST_NAME} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")`
  - 使用父 CMakeLists.txt 的 `"80;86;89"` 设置，而非硬编码 sm_86

  **6b. 迁移 CUDA 变量引用**:
  - 第 12 行：`${CUDA_INCLUDE_DIRS}` → `${CUDAToolkit_INCLUDE_DIRS}`
  - 第 15-17 行 `target_link_libraries`：替换
    - `${CUDA_LIBRARIES}` → `CUDA::cudart`
    - `${CUDA_cublas_LIBRARY}` → 删除（测试未使用 cuBLAS）
    - `${CUDA_cudart_LIBRARY}` → 删除（CUDA::cudart 替代）

  **Must NOT do**:
  - 不要删除测试文件或测试用例
  - 不要修改 `TEST_DATA_DIR` 编译定义（第 33-35 行）
  - 不要修改 `${TensorRT_LIBRARIES}` 引用

  **Recommended Agent Profile**:
  - **Category**: `quick` — 6 行修改，纯变量替换
  - **Reason**: CMake 变量名替换
  - **Skills**: 无特殊需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Task 6 (独立执行)
  - **Blocks**: Task 7（构建系统调优最后一步）
  - **Blocked By**: Task 5（依赖 CUDAToolkit 变量名）

  **References**:
  - `deploy/CPP/tests/CMakeLists.txt:12` — `${CUDA_INCLUDE_DIRS}` 替换点
  - `deploy/CPP/tests/CMakeLists.txt:14-17` — `target_link_libraries` 替换点
  - `deploy/CPP/tests/CMakeLists.txt:19` — 硬编码 `CUDA_ARCHITECTURES "86"`
  - `deploy/CPP/tests/CMakeLists.txt:33-35` — TEST_DATA_DIR 定义（不改）

  **Acceptance Criteria**:
  - [ ] 第 19 行使用 `${CMAKE_CUDA_ARCHITECTURES}` 非硬编码
  - [ ] 第 12 行使用 `${CUDAToolkit_INCLUDE_DIRS}`
  - [ ] 第 15-17 行链接 `CUDA::cudart` 而非已废弃变量

  **QA Scenarios**:

  ```
  Scenario: 测试 CMake 配置正确性（Linux 环境）
    Tool: Bash
    Steps:
      1. 上述 Task 5 配置成功后，检查 /tmp/test-build 中 tests 目录存在
      2. grep "CUDA_ARCHITECTURES" /tmp/test-build/tests/CMakeFiles/*.dir/flags.make
    Expected Result: 测试目标的架构列表与父 CMake 一致（80;86;89）
    Failure Indicators: 架构硬编码为 86
    Evidence: .omo/evidence/task-6-test-arch.txt

  Scenario: cuBLAS 依赖已移除
    Tool: Bash (grep)
    Steps:
      1. grep "cublas" deploy/CPP/tests/CMakeLists.txt
    Expected Result: 无输出（cuBLAS 引用已删除）
    Evidence: .omo/evidence/task-6-no-cublas.txt
  ```

  **Commit**: NO（合入最终 commit）
- [x] 7. `cmake/FindTensorRT.cmake` — TRT 10 DLL 命名适配 + Windows 路径优先

  **What to do**:
  **7a. 添加 TRT 10 库名搜索**:
  - 在第 79 行 `find_library(TensorRT_LIBRARY_NVINFER nvinfer ...)` 中，添加备选名称 `nvinfer_10`：
    ```cmake
    find_library(TensorRT_LIBRARY_NVINFER
      NAMES nvinfer nvinfer_10
      HINTS ${_TRT_SEARCH_PATHS}
      PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    )
    ```
  - 同理修改 `nvinfer_plugin`（第 85 行）→ 添加 `nvinfer_plugin_10`
  - 同理修改 `nvonnxparser`（第 91 行）→ 添加 `nvonnxparser_10`

  **7b. Windows 路径优先级提升**:
  - 将第 38-55 行（Linux/通用搜索路径）包裹在 `if(NOT WIN32)` 中
  - 将第 57-65 行（Windows 搜索路径）移到 `if(WIN32)` 块中，并放在最前面
  - 添加标准 Windows TensorRT 10 安装路径：
    ```cmake
    "C:/Program Files/NVIDIA Corporation/TensorRT-10.16.1"
    "C:/Program Files/NVIDIA Corporation/TensorRT-10.16.0"
    "C:/Program Files/NVIDIA Corporation/TensorRT"
    ```

  **7c. 版本注释更新**:
  - 第 2-5 行的模块文档注释：添加 TRT 10 支持说明

  **Must NOT do**:
  - 不要删除对旧名称 `nvinfer`（无 `_10` 后缀）的搜索（Linux TRT 8.6 兼容）
  - 不要修改 `find_package_handle_standard_args` 的版本检查逻辑
  - 不要使 `nvonnxparser` 可选（保持 REQUIRED）

  **Recommended Agent Profile**:
  - **Category**: `quick` — CMake 路径和名称调整，约 8 行
  - **Reason**: 配置文件修改
  - **Skills**: 无特殊需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Task 7 (独立执行)
  - **Blocks**: 无（最终任务）
  - **Blocked By**: Task 6

  **References**:
  - `deploy/CPP/cmake/FindTensorRT.cmake:38-55` — Linux/通用搜索路径
  - `deploy/CPP/cmake/FindTensorRT.cmake:57-65` — Windows 搜索路径
  - `deploy/CPP/cmake/FindTensorRT.cmake:79` — `nvinfer` 库搜索
  - `deploy/CPP/cmake/FindTensorRT.cmake:85` — `nvinfer_plugin` 库搜索
  - `deploy/CPP/cmake/FindTensorRT.cmake:91` — `nvonnxparser` 库搜索
  - `deploy/CPP/cmake/FindTensorRT.cmake:2-5` — 模块文档注释

  **Acceptance Criteria**:
  - [ ] `find_library` 搜索 `nvinfer` 和 `nvinfer_10`（三个库均如此）
  - [ ] Windows 搜索路径包含 `NVIDIA Corporation/TensorRT-10.*`
  - [ ] Linux 搜索路径（`/usr/local/TensorRT-*`）保持在 `if(NOT WIN32)` 块中

  **QA Scenarios**:

  ```
  Scenario: CMake 仍能在 Linux TRT 8.6 上找到库（回归检查）
    Tool: Bash
    Steps:
      1. cmake -B /tmp/test-findtrt -S deploy/CPP -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 2>&1
      2. grep "TensorRT version" 输出
    Expected Result: "TensorRT version: 8.6.1.6"
    Failure Indicators: "TensorRT not found" 或版本错误
    Evidence: .omo/evidence/task-7-linux-regression.txt

  Scenario: 库名包含 _10 后缀（语法检查）
    Tool: Bash (grep)
    Steps:
      1. grep "nvinfer_10\|nvinfer_plugin_10\|nvonnxparser_10" deploy/CPP/cmake/FindTensorRT.cmake
    Expected Result: 每个库名出现 1 次（共 3 行）
    Evidence: .omo/evidence/task-7-lib-names.txt
  ```

  **Commit**: NO（合入最终 commit）

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  逐项检查 "Must Have" 清单：UniquePtr 成员、CUDAToolkit 迁移、MSVC CRT 设置、FindTensorRT nvinfer_10 搜索、Linux 回归。搜索 "Must NOT Have" 禁止修改文件是否有变更。
  Output: `Must Have [5/5] | Must NOT Have [6/6] | Tasks [7/7] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  检查所有修改文件的代码质量：无 `as any`/`@ts-ignore`、空 catch、prod console.log、注释代码、未使用 import。验证 AI slop 模式（过度注释、过度抽象）。
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  顺序验证所有 QA 场景（从 Task 1 到 Task 7）。重点：trt_engine.h 类型检查、CMake 配置成功、测试编译通过。在 Linux 环境下执行完整回归。
  Output: `Scenarios [7/7 pass] | Regression [PASS/FAIL] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. 验证 1:1 — spec 中的全部 built（无遗漏），spec 外的全部 not built（无 scope creep）。检查 "Must NOT do" 合规。检测跨任务污染。
  Output: `Tasks [7/7 compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1-7**: 修复合入一个 commit
  - Message: `fix(deploy): migrate to TRT10 UniquePtr + CUDAToolkit + MSVC CRT`
  - Files: `deploy/CPP/include/trt_engine.h`, `deploy/CPP/src/trt_engine.cpp`, `deploy/CPP/include/trt_inference.h`, `deploy/CPP/src/trt_inference.cpp`, `deploy/CPP/CMakeLists.txt`, `deploy/CPP/tests/CMakeLists.txt`, `deploy/CPP/cmake/FindTensorRT.cmake`
  - Pre-commit: `ctest --test-dir deploy/CPP/build --output-on-failure`

---

## Success Criteria

### Verification Commands
```bash
# Linux 回归
cmake -B deploy/CPP/build_linux -S deploy/CPP -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6
cmake --build deploy/CPP/build_linux -j$(nproc)
ctest --test-dir deploy/CPP/build_linux --output-on-failure

# Windows 构建 (目标环境，如环境可用)
cmake -B deploy/CPP/build_win -S deploy/CPP -G "Visual Studio 17 2022" -A x64 -DTensorRT_ROOT="C:/TensorRT-10.x.x.x"
cmake --build deploy/CPP/build_win --config Release
```

### Final Checklist
- [ ] 所有 "Must Have" present
- [ ] 所有 "Must NOT Have" absent
- [ ] Linux 回归编译成功 + ctest 3/3 pass
- [ ] Windows CMake 配置无 CMP0146/FindCUDA 错误（模拟）
