# Oracle 审查修复：P0 CMAKE_RUNTIME_OUTPUT_DIRECTORY + TensorRT_ROOT

## TL;DR

> **Quick Summary**: 修复 Oracle 审查发现的两个 P0 问题：Windows DLL 部署目标路径为空、Windows 上 TensorRT_ROOT 被忽略。
>
> **Deliverables**:
> - CMakeLists.txt: 添加 `CMAKE_RUNTIME_OUTPUT_DIRECTORY` 后备值
> - FindTensorRT.cmake: Windows 分支添加 `TensorRT_ROOT` 优先搜索
>
> **Estimated Effort**: Quick（2 个文件，共 5 行修改）
> **Parallel Execution**: YES — 两个修复独立，可并行执行

---

## Context

### Oracle 审查发现

Oracle agent 对 `cpp-deploy-windows-fix` 计划的 7 个修改文件进行全面审查，确认 RAII 管理、所有权模型、异常安全全部正确。但发现两个 P0 级别问题：

1. **CMAKE_RUNTIME_OUTPUT_DIRECTORY 可为空** — Visual Studio 多配置生成器下默认未设置，导致 `"/"` 作为 DLL 复制目标
2. **Windows 上 TensorRT_ROOT 被忽略** — `if(WIN32)` 分支无条件填充硬编码路径，忽略了用户通过 `-DTensorRT_ROOT=...` 指定的自定义路径

### Oracle 建议修复方案

- P0-1：在 `foreach(DLL ...)` 前添加 `if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY) set(...) endif()`
- P0-2：在 `WIN32` 分支的硬编码路径后，添加 `if(TensorRT_ROOT) list(PREPEND _TRT_SEARCH_PATHS "${TensorRT_ROOT}") endif()`

---

## Work Objectives

### Core Objective
修复 Oracle 审查发现的两个 P0 问题，确保 Windows + VS 2022 构建正确，且用户可通过 TensorRT_ROOT 指定自定义路径。

### Concrete Deliverables
- `deploy/CPP/CMakeLists.txt` — 添加 `CMAKE_RUNTIME_OUTPUT_DIRECTORY` 后备值
- `deploy/CPP/cmake/FindTensorRT.cmake` — Windows 分支添加 TensorRT_ROOT 优先搜索

### Must NOT Have (Guardrails)
- 不修改 CUDA kernel、推理逻辑、main.cpp、tinyply.h
- 不改变 Linux 构建行为

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES（CMake 配置验证）
- **Automated tests**: 无（CMake 语法检查）
- **Agent-Executed QA**: 语法验证 + grep 检查

---

## Execution Strategy

### Parallel Execution（两个修复独立）

```
Task 1 (CMakeLists.txt) ──┐
                           ├── 可并行
Task 2 (FindTensorRT.cmake)┘
```

---

## TODOs

- [ ] 1. `CMakeLists.txt` — 添加 CMAKE_RUNTIME_OUTPUT_DIRECTORY 后备值

  **What to do**:
  - 在第 73 行 `elseif(WIN32)` 块内，`file(GLOB ...)` 之前，插入：
    ```cmake
    if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif()
    ```
  - 确保不修改 `if(UNIX)` RPATH 块
  - 确保不修改 `foreach(DLL ...)` 之后的逻辑

  **Must NOT do**:
  - 不要修改 RPATH 配置
  - 不要修改 DLL GLOB 或 foreach 逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick` — 3 行插入，纯 CMake 语法
  - **Reason**: 简单条件判断 + set，无逻辑变更

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Task 2 并行）
  - **Blocks**: 无
  - **Blocked By**: 无

  **References**:
  - `deploy/CPP/CMakeLists.txt:72-79` — `elseif(WIN32)` 块，插入点在第 73 行后

  **Acceptance Criteria**:
  - [ ] `elseif(WIN32)` 块内首行有 `if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY) set(...) endif()`
  - [ ] 语法检查：`cmake -P deploy/CPP/CMakeLists.txt` 无报错（注：-P 不适用，改为 grep 验证）

  **QA Scenarios**:

  ```
  Scenario: 后备值存在性检查（Happy path）
    Tool: Bash (grep)
    Steps:
      1. grep -A2 "elseif(WIN32)" deploy/CPP/CMakeLists.txt
    Expected Result: 输出包含 "if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)" 和 "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY"
    Failure Indicators: grep 无输出或不包含后备值设置
    Evidence: .omo/evidence/task-r1-guard.txt

  Scenario: RPATH 块未被修改
    Tool: Bash (grep)
    Steps:
      1. grep -A5 "if(UNIX)" deploy/CPP/CMakeLists.txt
    Expected Result: 原始 RPATH 配置不变
    Failure Indicators: RPATH 块被意外修改
    Evidence: .omo/evidence/task-r1-rpath.txt
  ```

  **Commit**: NO（合入最终 commit）

- [ ] 2. `cmake/FindTensorRT.cmake` — Windows 分支添加 TensorRT_ROOT 优先搜索

  **What to do**:
  - 在第 44 行（硬编码 Windows 路径块的 `)` 之后，`elseif(TensorRT_ROOT)` 之前），插入：
    ```cmake
    # 用户指定的 TensorRT_ROOT 优先于硬编码路径
    if(TensorRT_ROOT)
      list(PREPEND _TRT_SEARCH_PATHS "${TensorRT_ROOT}")
    endif()
    ```
  - 确保 `elseif(TensorRT_ROOT)` 分支不变
  - 确保 Linux `else()` 分支不变

  **Must NOT do**:
  - 不要修改 Linux 搜索路径分支
  - 不要修改 `find_package_handle_standard_args` 逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick` — 4 行插入，纯 CMake 语法
  - **Reason**: 简单条件判断 + list 操作

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Task 1 并行）
  - **Blocks**: 无
  - **Blocked By**: 无

  **References**:
  - `deploy/CPP/cmake/FindTensorRT.cmake:34-45` — Windows 路径块 + `elseif(TensorRT_ROOT)` 边界

  **Acceptance Criteria**:
  - [ ] `if(WIN32)` 分支末尾有 `if(TensorRT_ROOT) list(PREPEND ...) endif()`
  - [ ] `elseif(TensorRT_ROOT)` 分支在插入后位置不变
  - [ ] Linux 搜索路径不变

  **QA Scenarios**:

  ```
  Scenario: TensorRT_ROOT 被 list(PREPEND) 插入（Happy path）
    Tool: Bash (grep)
    Steps:
      1. grep -A3 'if(TensorRT_ROOT)' deploy/CPP/cmake/FindTensorRT.cmake | head -10
    Expected Result: 在 Windows 路径块后看到 "if(TensorRT_ROOT)" + "list(PREPEND ... ${TensorRT_ROOT}"
    Failure Indicators: 未找到 PREPEND 或在错误位置
    Evidence: .omo/evidence/task-r2-prepend.txt

  Scenario: Linux 路径未被影响
    Tool: Bash (grep)
    Steps:
      1. grep "elseif(TensorRT_ROOT)" deploy/CPP/cmake/FindTensorRT.cmake
      2. grep -c "elseif(TensorRT_ROOT)" deploy/CPP/cmake/FindTensorRT.cmake
    Expected Result: 出现 2 次（一次在 WIN32 块内，一次在后续分支）
    Failure Indicators: count != 2 或 Linux 分支被意外修改
    Evidence: .omo/evidence/task-r2-branches.txt
  ```

  **Commit**: NO（合入最终 commit）

---

## Commit Strategy

- **1-2**: 修复合入一个 commit
  - Message: `fix(deploy): guard CMAKE_RUNTIME_OUTPUT_DIRECTORY + prepend TensorRT_ROOT on Windows`
  - Files: `deploy/CPP/CMakeLists.txt`, `deploy/CPP/cmake/FindTensorRT.cmake`

---

## Success Criteria

### Verification Commands
```bash
# P0-1 验证
grep -A3 "elseif(WIN32)" deploy/CPP/CMakeLists.txt | grep "CMAKE_RUNTIME_OUTPUT_DIRECTORY"

# P0-2 验证
grep -A3 "if(TensorRT_ROOT)" deploy/CPP/cmake/FindTensorRT.cmake | grep "list(PREPEND"
grep -c "elseif(TensorRT_ROOT)" deploy/CPP/cmake/FindTensorRT.cmake
```
