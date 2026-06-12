# UniquePtr 回退 + Oracle P0 修复

## TL;DR

> **Quick Summary**: 回退 4 个 C++ 文件的 `nvinfer1::UniquePtr` 修改（此类型不存在于 TRT API），恢复原始裸指针 + `delete` 方式（已是 TRT 10 兼容）。保留 CMake 构建系统修改。附加 Oracle 发现的 2 个 P0 修复。
>
> **Deliverables**:
> - 4 个 C++ 文件回退至原始状态（裸指针 + delete）
> - CMakeLists.txt: 追加 `CMAKE_RUNTIME_OUTPUT_DIRECTORY` 后备值（Oracle P0-1）
> - FindTensorRT.cmake: 追加 `TensorRT_ROOT` 优先搜索（Oracle P0-2）
>
> **Estimated Effort**: Quick（6 个文件修改，~20 行）
> **Parallel Execution**: YES — 4 个 C++ 回退可并行，2 个 CMake 修复可并行

---

## Context

### 发现

用户指出 `nvinfer1::UniquePtr` 不存在于 TensorRT 官方 C++ API。经查证：

- **TRT 8.6.1.6 头文件**（当前环境）：`grep -rn "UniquePtr" /usr/local/TensorRT-8.6.1.6/include/` 结果为空
- **TRT 10 官方迁移指南**：
  - `IRuntime::destroy()` → `delete ObjectName`
  - `ICudaEngine::destroy()` → `delete ObjectName`
  - `IExecutionContext::destroy()` → `delete ObjectName`
  - 工厂函数 `createInferRuntime()`、`deserializeCudaEngine()`、`createExecutionContext()` 返回的仍是**裸指针**
- **TRT 10 示例中**用的 `SampleUniquePtr` 是 `samples/common/` 中的工具类，不属于 `nvinfer1` 命名空间
- **TRT 8.6 已有 virtual 析构函数**：`virtual ~IRuntime() noexcept = default;` → `delete` 已是合法释放方式

### 为什么原始代码已经正确

原始代码使用：
- 裸指针成员变量（`IRuntime*`, `ICudaEngine*`, `IExecutionContext*`）
- 析构函数中 `delete` 释放
- 工厂函数直接赋值裸指针

这与 TRT 10 的 API 完全兼容——`delete` 就是 TRT 10 的标准释放方式。原生 C++ 代码无需修改。

### 需要回退的修改

| 文件 | 错误修改 | 正确状态 |
|------|---------|---------|
| `trt_engine.h` | `UniquePtr<IRuntime>` / `UniquePtr<ICudaEngine>` | `IRuntime* runtime_ = nullptr` / `ICudaEngine* engine_ = nullptr` |
| `trt_engine.cpp` | 删除手动 delete / 添加 .release() | 恢复 delete engine_/runtime_ / 恢复直接 return engine_->createExecutionContext() |
| `trt_inference.h` | `UniquePtr<IExecutionContext>` | `IExecutionContext* context_ = nullptr` |
| `trt_inference.cpp` | .reset() 接管 / 删除手动 delete | 直接赋值 context_ = engine_->create_context() / 恢复 delete context_ |

### 保留的修改（正确）

| 文件 | 修改 | 状态 |
|------|------|------|
| `CMakeLists.txt` | CUDAToolkit + MSVC CRT + DLL 部署 | ✅ 保留 |
| `tests/CMakeLists.txt` | 变量迁移、CUDA_ARCHITECTURES 继承 | ✅ 保留 |
| `FindTensorRT.cmake` | TRT 10 DLL 命名 + Windows 路径 | ✅ 保留 |

### 附加修复（Oracle P0）

| # | 问题 | 位置 | 修复 |
|---|------|------|------|
| P0-1 | `CMAKE_RUNTIME_OUTPUT_DIRECTORY` 在 VS 下为空 | `CMakeLists.txt:73` | 添加 `if(NOT ...) set(... "${CMAKE_BINARY_DIR}")` |
| P0-2 | Windows 上用户指定的 `TensorRT_ROOT` 被忽略 | `FindTensorRT.cmake:44` | 添加 `list(PREPEND ...)` |

---

## Work Objectives

### Core Objective
回退 4 个 C++ 文件的 `nvinfer1::UniquePtr` 修改，恢复原始裸指针管理模式，同时保留正确的 CMake 修改并附加 Oracle P0 修复。

### Concrete Deliverables
- `deploy/CPP/include/trt_engine.h` — 恢复裸指针成员 + `.get()` → 裸指针直接返回
- `deploy/CPP/src/trt_engine.cpp` — 恢复手动 delete + 移除 .release()
- `deploy/CPP/include/trt_inference.h` — 恢复裸指针 context_ + `.get()` → 裸指针直接返回
- `deploy/CPP/src/trt_inference.cpp` — 恢复直接赋值 + 恢复手动 delete
- `deploy/CPP/CMakeLists.txt` — 追加 CMAKE_RUNTIME_OUTPUT_DIRECTORY 后备值
- `deploy/CPP/cmake/FindTensorRT.cmake` — 追加 TensorRT_ROOT 优先搜索

### Must Have
- 原始 `delete` 行为恢复（析构函数中 `delete engine_` / `delete runtime_` / `delete context_`）
- `create_context()` 返回 `engine_->createExecutionContext()`（裸指针，不含 `.release()`）
- `get()` / `get_context()` 直接返回裸指针成员（不含 `.get()`）
- CMake 的 CUDAToolkit / MSVC CRT / DLL 部署修改保留

### Must NOT Have
- 不得残留任何 `nvinfer1::UniquePtr` 引用
- 不得残留任何 `.release()` 或 `.reset()` 或 `.get()` 调用（针对 TRT 对象的）
- 不得删除 `#include <NvInfer.h>` 也不得添加不必要的 include
- 不得修改 CMake 文件中 CUDAToolkit / MSVC CRT / DLL 的正确部分

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: 无（C++ 编译验证依赖 TRT 头文件）
- **Automated tests**: 无
- **Agent-Executed QA**: grep 验证 + git diff 验证

### QA Policy
- 每个任务完成后用 grep 验证目标模式存在、禁止模式不存在
- 回退完成后 git diff 对比确认净变化仅为 CMake 修改

---

## Execution Strategy

### Parallel Execution（两组并行）

```
Wave 1（可同步执行，互不依赖）:
├── Task 1: trt_engine.h 回退
├── Task 2: trt_engine.cpp 回退
├── Task 3: trt_inference.h 回退
└── Task 4: trt_inference.cpp 回退

Wave 2（CMake 修复，也可同步）:
├── Task 5: CMakeLists.txt P0-1 修复
└── Task 6: FindTensorRT.cmake P0-2 修复
```

---

## TODOs

- [x] 1. `trt_engine.h` — 回退 UniquePtr → 裸指针

  **What to do**:
  - 第 39 行：`return engine_.get();` → 恢复为 `return engine_;`
  - 第 57 行：`nvinfer1::UniquePtr<nvinfer1::IRuntime> runtime_;` → 恢复为 `nvinfer1::IRuntime* runtime_ = nullptr;`
  - 第 58 行：`nvinfer1::UniquePtr<nvinfer1::ICudaEngine> engine_;` → 恢复为 `nvinfer1::ICudaEngine* engine_ = nullptr;`

  **Must NOT do**:
  - 不修改 `#include <NvInfer.h>`
  - 不修改其他任何方法声明

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 3 行替换，纯机械恢复

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Task 2, 3, 4 并行）

  **Acceptance Criteria**:
  - [ ] `runtime_` 类型为 `nvinfer1::IRuntime*`
  - [ ] `engine_` 类型为 `nvinfer1::ICudaEngine*`
  - [ ] `get()` 返回 `engine_`（不含 `.get()`）

  **QA Scenarios**:
  ```
  Scenario: 裸指针恢复验证
    Tool: Bash (grep)
    Steps:
      1. grep "IRuntime\* runtime_" deploy/CPP/include/trt_engine.h
      2. grep "ICudaEngine\* engine_" deploy/CPP/include/trt_engine.h
      3. grep "return engine_;" deploy/CPP/include/trt_engine.h
    Expected Result: 全部命中，无 UniquePtr 残留
    Evidence: .omo/evidence/task-u1-revert.txt
  ```

- [x] 2. `trt_engine.cpp` — 回退 delete + 移除 .release()

  **What to do**:
  - 第 26-29 行：删除新增的 "TRT 10.x 中所有 TRT 对象使用 UniquePtr..." 注释块（恢复为原始注释）
  - 第 91-93 行：`TrEngine::~TrEngine()` — 从空函数体恢复为：
    ```cpp
    if (engine_) { delete engine_; engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
    ```
  - 第 119-121 行：`return engine_->createExecutionContext().release();` → 恢复为 `return engine_->createExecutionContext();`（删除 `.release()` 调用和前置注释）

  **Must NOT do**:
  - 不修改构造函数中的工厂函数调用（第 56、63 行保持不变）
  - 不修改 IO 遍历逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 3 处恢复，纯机械操作

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Task 1, 3, 4 并行）

  **Acceptance Criteria**:
  - [ ] 析构函数中有 `delete engine_` 和 `delete runtime_`
  - [ ] `create_context()` 返回 `engine_->createExecutionContext()`（不含 `.release()`）
  - [ ] 无 UniquePtr 相关注释

  **QA Scenarios**:
  ```
  Scenario: delete 语句恢复验证
    Tool: Bash (grep)
    Steps:
      1. grep "delete engine_" deploy/CPP/src/trt_engine.cpp
      2. grep "delete runtime_" deploy/CPP/src/trt_engine.cpp
      3. grep "\.release()" deploy/CPP/src/trt_engine.cpp
    Expected Result: 前两条命中，第三条无结果
    Evidence: .omo/evidence/task-u2-revert.txt
  ```

- [x] 3. `trt_inference.h` — 回退 UniquePtr → 裸指针

  **What to do**:
  - 第 67 行：`return context_.get();` → 恢复为 `return context_;`
  - 第 75 行：`nvinfer1::UniquePtr<nvinfer1::IExecutionContext> context_;` → 恢复为 `nvinfer1::IExecutionContext* context_ = nullptr;`

  **Must NOT do**:
  - 不修改 `#include <NvInfer.h>`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 2 行替换

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Task 1, 2, 4 并行）

  **Acceptance Criteria**:
  - [ ] `context_` 类型为 `nvinfer1::IExecutionContext*`
  - [ ] `get_context()` 返回 `context_`（不含 `.get()`）

  **QA Scenarios**:
  ```
  Scenario: 裸指针恢复验证
    Tool: Bash (grep)
    Steps:
      1. grep "IExecutionContext\* context_" deploy/CPP/include/trt_inference.h
      2. grep "return context_;" deploy/CPP/include/trt_inference.h
    Expected Result: 全部命中
    Evidence: .omo/evidence/task-u3-revert.txt
  ```

- [x] 4. `trt_inference.cpp` — 回退 .reset() + 恢复 delete

  **What to do**:
  - 第 15-16 行：`context_.reset(engine_->create_context());` → 恢复为 `context_ = engine_->create_context();`（含原始注释）
  - 第 33-35 行：`TrInference::~TrInference()` — 从空函数体恢复为：
    ```cpp
    if (context_) { delete context_; context_ = nullptr; }
    ```

  **Must NOT do**:
  - 不修改 `CHECK_TRT(context_, ...)` 调用
  - 不修改 `engine_->get()` 调用（`engine_` 仍是 `TrEngine*` 裸指针）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Reason**: 2 处恢复

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Task 1, 2, 3 并行）

  **Acceptance Criteria**:
  - [ ] 构造函数中 `context_ = engine_->create_context();`（不含 `.reset()`）
  - [ ] 析构函数中有 `delete context_`

  **QA Scenarios**:
  ```
  Scenario: 原始赋值和 delete 恢复验证
    Tool: Bash (grep)
    Steps:
      1. grep "context_ = engine_->create_context()" deploy/CPP/src/trt_inference.cpp
      2. grep "\.reset(" deploy/CPP/src/trt_inference.cpp
      3. grep "delete context_" deploy/CPP/src/trt_inference.cpp
    Expected Result: 第1条命中，第2条无结果，第3条命中
    Evidence: .omo/evidence/task-u4-revert.txt
  ```

- [x] 5. `CMakeLists.txt` — Oracle P0-1: CMAKE_RUNTIME_OUTPUT_DIRECTORY 后备值

  **What to do**:
  - 在 `elseif(WIN32)` 块内第 73 行后，`file(GLOB ...)` 之前，插入：
    ```cmake
    if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif()
    ```

  **Must NOT do**:
  - 不修改 `if(UNIX)` RPATH 块
  - 不修改 CUDAToolkit / MSVC CRT 配置

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Task 6 并行）

  **Acceptance Criteria**:
  - [ ] `elseif(WIN32)` 内首行有 `if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY) set(...) endif()`

  **QA Scenarios**:
  ```
  Scenario: 后备值存在性验证
    Tool: Bash (grep)
    Steps:
      1. grep -A2 "elseif(WIN32)" deploy/CPP/CMakeLists.txt | grep "CMAKE_RUNTIME_OUTPUT_DIRECTORY"
    Expected Result: 命中
    Evidence: .omo/evidence/task-u5-guard.txt
  ```

- [x] 6. `cmake/FindTensorRT.cmake` — Oracle P0-2: TensorRT_ROOT 优先搜索

  **What to do**:
  - 在第 44 行（硬编码 Windows 路径块末尾 `)` 之后，`elseif(TensorRT_ROOT)` 之前），插入：
    ```cmake
    # 用户指定的 TensorRT_ROOT 优先于硬编码路径
    if(TensorRT_ROOT)
      list(PREPEND _TRT_SEARCH_PATHS "${TensorRT_ROOT}")
    endif()
    ```

  **Must NOT do**:
  - 不修改 Linux 搜索路径分支
  - 不修改 `find_package_handle_standard_args` 逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Task 5 并行）

  **Acceptance Criteria**:
  - [ ] `if(WIN32)` 分支末尾有 `if(TensorRT_ROOT) list(PREPEND ...) endif()`

  **QA Scenarios**:
  ```
  Scenario: PREPEND 插入验证
    Tool: Bash (grep)
    Steps:
      1. grep -A3 "if(TensorRT_ROOT)" deploy/CPP/cmake/FindTensorRT.cmake
    Expected Result: 在 Windows 路径块后看到 "list(PREPEND ... ${TensorRT_ROOT}"
    Evidence: .omo/evidence/task-u6-prepend.txt
  ```

---

## Commit Strategy

- **全部 6 个修改合入一个 commit**
  - Message: `fix(deploy): revert UniquePtr to raw pointers + Oracle P0 fixes`
  - Files: `deploy/CPP/include/trt_engine.h`, `deploy/CPP/src/trt_engine.cpp`, `deploy/CPP/include/trt_inference.h`, `deploy/CPP/src/trt_inference.cpp`, `deploy/CPP/CMakeLists.txt`, `deploy/CPP/cmake/FindTensorRT.cmake`

---

## Success Criteria

### Verification Commands
```bash
# 确认无 UniquePtr 残留
grep -rn "UniquePtr" deploy/CPP/include/ deploy/CPP/src/

# 确认 delete 语句恢复
grep -n "delete engine_\|delete runtime_\|delete context_" deploy/CPP/src/trt_engine.cpp deploy/CPP/src/trt_inference.cpp

# 确认无 .release() / .reset() 残留
grep -rn "\.release()\|context_\.reset" deploy/CPP/src/

# 确认 CMake 正确修改保留
grep "CUDAToolkit" deploy/CPP/CMakeLists.txt
grep "MultiThreadedDLL" deploy/CPP/CMakeLists.txt
grep "nvinfer_10" deploy/CPP/cmake/FindTensorRT.cmake

# 净变化：仅 CMake + FindTensorRT 有变更
git diff --stat HEAD -- deploy/CPP/
```
