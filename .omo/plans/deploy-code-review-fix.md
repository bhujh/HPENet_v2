# Deploy Code Review Fixes

## TL;DR

> **Quick Summary**: Fix 1 parameter default mismatch, remove 1 unused function, extract shared preprocessing code into `deploy/common.py` to eliminate ~100 lines of duplication.
> 
> **Deliverables**:
> - `deploy/common.py` — 共享的数据加载+预处理+归一化函数
> - `deploy/trt_utils.py` — 删除 `_is_cuda_available`
> - `deploy/trt_inference.py` — 修复 `min_n=64→1024` 默认值，import common
> - `deploy/onnx_inference.py` — 删除重复函数，import common
> 
> **Estimated Effort**: Quick (3 tasks, all parallel)
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: None (all independent)

---

## Context

### Original Request
对 deploy 目录下的部署源码进行代码审查，修复发现的问题。

### Interview Summary
**Key Discussions**:
- 审查确认：预处理、算子替换、checkpoint 路径全部正确
- TRT FP32 已实际运行验证，与 ONNX/PyTorch 100% 预测一致
- 发现 1 个需修复问题（min_n 默认值）+ 2 个非阻塞建议

**Research Findings**:
- `trt_inference.py:157` 函数 `min_n` 默认值 64 与实际 TRT engine 构建时的 1024 不一致
- `trt_utils.py:45-51` `_is_cuda_available()` 定义了但从未被调用
- `onnx_inference.py` 和 `trt_inference.py` 中有约 100 行相同的函数

---

## Work Objectives

### Core Objective
修复部署代码中的 3 个问题，消除代码重复，保持与已验证版本的功能等价。

### Concrete Deliverables
- `deploy/common.py` — 新增，包含 `load_data_ply`, `preprocess_test`, `load_stats`, `preprocess_subcloud`
- `deploy/trt_utils.py` — 删除 `_is_cuda_available()` 函数
- `deploy/trt_inference.py` — `min_n=64→1024`，删除重复函数改为 import
- `deploy/onnx_inference.py` — 删除重复函数改为 import

### Must Have
- 功能完全等价（predictions 100% 不变）
- `python deploy/onnx_inference.py --num_files 3` 正常运行
- `python deploy/trt_inference.py --num_files 3` 正常运行

### Must NOT Have (Guardrails)
- 不修改任何模型逻辑或算子替换代码
- 不修改 ONNX export / TRT build 流程
- 不修改 `deploy/onnx_backend.py`（已验证正确）

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: NO（手工运行验证）
- **Agent-Executed QA**: YES

### QA Policy
每个任务完成后运行对应的推理脚本，对比输出与修复前一致。

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (MAX PARALLEL — 3 tasks, all independent):
├── Task 1: deploy/common.py — 提取共享函数
├── Task 2: deploy/trt_utils.py — 删除 _is_cuda_available
└── Task 3: deploy/trt_inference.py — 修复 min_n 默认值

Wave 2 (after Wave 1 — 2 tasks):
├── Task 4: deploy/onnx_inference.py — 删除重复、导入 common
└── Task 5: deploy/trt_inference.py — 删除重复、导入 common

Wave FINAL:
├── F1: 运行 onnx_inference.py 验证
├── F2: 运行 trt_inference.py 验证
└── F3: diff 对比输出一致性
```

---

## TODOs

- [x] 1. 创建 `deploy/common.py` — 提取共享函数

  **What to do**:
  - 从 `trt_inference.py` 复制以下函数到新文件 `deploy/common.py`：
    - `load_data_ply(data_path)` — PLY 文件读取
    - `preprocess_test(coord, feat, voxel_size=0.1)` — 体素化 + 子点云切分
    - `load_stats(stats_file)` — 特征统计量加载
    - `preprocess_subcloud(coord, feat, idx_part, feat_mean, feat_std, z_mean, z_std, gravity_dim=2)` — 单子点云预处理
  - 添加 `sys.path.insert(0, ...)` 确保 `openpoints` 可导入
  - 添加必要的 import：`os`, `sys`, `numpy as np`, `torch`, `plyfile.PlyData`
  - `from openpoints.dataset.data_util import voxelize`

  **Must NOT do**:
  - 不要修改任何函数的逻辑或签名
  - 不要引入额外的依赖

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 纯代码搬运，无逻辑变更

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: None

  **Acceptance Criteria**:
  - `deploy/common.py` 文件存在
  - 包含全部 4 个函数
  - `python -c "from deploy.common import load_data_ply"` 无报错

  **QA Scenarios**:
  ```
  Scenario: import deploy/common module
    Tool: Bash
    Preconditions: conda activate hpenet
    Steps:
      1. python -c "from deploy.common import load_data_ply, preprocess_test, load_stats, preprocess_subcloud"
      2. python -c "from deploy.common import load_data_ply; print(load_data_ply.__doc__)"
    Expected Result: 无 import 错误，函数可调用
    Evidence: .omo/evidence/task-1-import-ok.txt

  Scenario: load_data_ply works on real file
    Tool: Bash
    Preconditions: conda activate hpenet, data/RadarClassi/radarfull/raw/0000001.ply exists
    Steps:
      1. python -c "from deploy.common import load_data_ply; c,f,l = load_data_ply('data/RadarClassi/radarfull/raw/0000001.ply'); print(c.shape, f.shape, l.shape)"
      2. Assert output contains shapes like "(N, 3) (N, 3) (N,)"
    Expected Result: 正常输出点的数量和维度
    Evidence: .omo/evidence/task-1-load-data.txt
  ```

- [x] 2. `deploy/trt_utils.py` — 删除 `_is_cuda_available()`

  **What to do**:
  - 删除第 45-51 行的 `_is_cuda_available()` 函数定义
  - 删除第 14 行的 `import ctypes`（仅被 `_is_cuda_available` 使用）

  **Must NOT do**:
  - 不要修改其他任何代码

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 单文件两处删除

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: None
  - **Blocked By**: None

  **Acceptance Criteria**:
  - `grep "_is_cuda_available" deploy/trt_utils.py` 无输出
  - `grep "import ctypes" deploy/trt_utils.py` 无输出
  - `python -c "from deploy.trt_utils import setup_trt_env, TRTSession"` 无报错

  **QA Scenarios**:
  ```
  Scenario: imports still work after removal
    Tool: Bash
    Preconditions: conda activate hpenet, LD_LIBRARY_PATH set
    Steps:
      1. LD_LIBRARY_PATH=/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:/usr/local/cuda-11.8/lib64 python -c "from deploy.trt_utils import setup_trt_env, TRTSession; print('OK')"
    Expected Result: prints "OK"
    Evidence: .omo/evidence/task-2-import-ok.txt
  ```

- [x] 3. `deploy/trt_inference.py:157` — 修复 `min_n` 默认值

  **What to do**:
  - 将 `infer_one_cloud_trt` 函数签名中的 `min_n=64` 改为 `min_n=1024`
  - （第 157 行）

  **Must NOT do**:
  - 不要修改函数体

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 单行修改

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **Acceptance Criteria**:
  - `grep "min_n=64" deploy/trt_inference.py` 无输出
  - `grep "min_n=1024" deploy/trt_inference.py` 有两处匹配 (函数签名 + argparse default)

  **QA Scenarios**:
  ```
  Scenario: verify function signature
    Tool: Bash
    Steps:
      1. grep "def infer_one_cloud_trt" deploy/trt_inference.py
    Expected Result: shows min_n=1024 in the signature
    Evidence: .omo/evidence/task-3-signature.txt
  ```

- [x] 4. `deploy/onnx_inference.py` — 删除重复函数，从 common 导入

  **What to do**:
  - 删除以下函数定义（它们的实现与 `common.py` 完全一致）：
    - `load_data_ply`（第 33-59 行）
    - `preprocess_test`（第 62-89 行）
    - `load_stats`（第 96-104 行）
  - 在文件顶部添加 `from deploy.common import load_data_ply, preprocess_test, load_stats`
  - 删除不再需要的 import：`plyfile.PlyData`（如果仅用于 load_data_ply）
  - `load_data_ply` 仅被 main() 调用，`preprocess_test` 仅被 main() 调用，`load_stats` 仅被 main() 调用 — 确认无误

  **Must NOT do**:
  - 不要删除 `run_onnx_inference`、`run_pytorch_inference`、`infer_one_cloud_*` — 这些不重复
  - 不要修改 `preprocess_subcloud` 的实现逻辑（如果 onnx_inference 中有内联版本）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 删除 + 添加 import

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 5)
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **QA Scenarios**:
  ```
  Scenario: ONNX inference still works
    Tool: Bash
    Preconditions: deploy/onnx_model.onnx exists, conda activate hpenet
    Steps:
      1. python deploy/onnx_inference.py --num_files 1 2>&1
    Expected Result: prints acc value, no import errors
    Failure Indicators: ModuleNotFoundError, NameError
    Evidence: .omo/evidence/task-4-onnx-ok.txt
  ```

- [x] 5. `deploy/trt_inference.py` — 删除重复函数，从 common 导入

  **What to do**:
  - 删除以下函数定义：
    - `load_data_ply`（第 39-65 行）
    - `preprocess_test`（第 68-89 行）
    - `load_stats`（第 92-100 行）
    - `preprocess_subcloud`（第 103-130 行）
  - 在文件顶部添加 `from deploy.common import load_data_ply, preprocess_test, load_stats, preprocess_subcloud`
  - 删除不再需要的 import：`plyfile.PlyData`、`torch`（如果仅用于被删除函数 — 确认 torch 仍被 `infer_one_cloud_trt` 使用，保留）
  - 保留 `pad_subcloud`（仅在 trt_inference.py 中使用，是 TRT 特有的 padding 逻辑）

  **Must NOT do**:
  - 不要删除 `pad_subcloud` — 这是 TRT 特有的
  - 不要修改 `infer_one_cloud_trt/onnx/pytorch` 中的逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 删除 + 添加 import

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: F2
  - **Blocked By**: Tasks 1, 3

  **QA Scenarios**:
  ```
  Scenario: TRT inference still works
    Tool: Bash
    Preconditions: deploy/trt_model_fp32.engine exists, conda activate hpenet
    Steps:
      1. LD_LIBRARY_PATH=/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:/usr/local/cuda-11.8/lib64 python deploy/trt_inference.py --num_files 1 2>&1
    Expected Result: prints acc value, no import errors
    Failure Indicators: ModuleNotFoundError, NameError, CUDA error
    Evidence: .omo/evidence/task-5-trt-ok.txt
  ```

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [x] F1. **ONNX 推理回归测试** — `quick`
- [x] F2. **TRT 推理回归测试** — `quick`
- [x] F3. **代码清洁度检查** — `quick`
  确认:
  - `grep "def load_data_ply" deploy/onnx_inference.py deploy/trt_inference.py` 无输出（已删除）
  - `grep "def load_data_ply" deploy/common.py` 有输出（已提取）
  - `grep "_is_cuda_available" deploy/trt_utils.py` 无输出
  - `grep "min_n=64" deploy/trt_inference.py` 无输出

---

## Commit Strategy

- **1-3**: `refactor(deploy): extract common.py, remove dead code, fix min_n default`
  - `deploy/common.py` (new)
  - `deploy/trt_utils.py` (remove _is_cuda_available + ctypes import)
  - `deploy/trt_inference.py` (min_n=1024)

- **4-5**: `refactor(deploy): import common in inference scripts`
  - `deploy/onnx_inference.py` (remove duplicate functions, import common)
  - `deploy/trt_inference.py` (remove duplicate functions, import common)

---

## Success Criteria

### Verification Commands
```bash
# ONNX
python deploy/onnx_inference.py --num_files 3 --compare

# TRT
LD_LIBRARY_PATH=/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:/usr/local/cuda-11.8/lib64 \
python deploy/trt_inference.py --num_files 3 --compare --engine deploy/trt_model_fp32.engine
```

### Final Checklist
- [ ] `deploy/common.py` 存在，4 个函数可导入
- [ ] `onnx_inference.py` 从 common 导入，无 `def load_data_ply` 等重复定义
- [ ] `trt_inference.py` 从 common 导入，无 `def load_data_ply` 等重复定义
- [ ] `trt_utils.py` 不包含 `_is_cuda_available` 和 `import ctypes`
- [ ] `trt_inference.py` 中 `infer_one_cloud_trt` 的 `min_n=1024`
- [ ] 两个推理脚本运行结果与修复前一致

---

## Commit Strategy

- **1-3**: `refactor(deploy): extract shared preprocessing to common.py` — 3 files
- **4-5**: `fix(deploy): remove unused code and fix default parameter` — 2 files

---

## Success Criteria

### Verification Commands
```bash
python deploy/onnx_inference.py --num_files 3
python deploy/trt_inference.py --num_files 3 --engine deploy/trt_model_fp32.engine
```

### Final Checklist
- [ ] `deploy/common.py` 存在，包含所有共享函数
- [ ] `onnx_inference.py` 和 `trt_inference.py` 从 common 导入
- [ ] `trt_utils.py` 不包含 `_is_cuda_available`
- [ ] `trt_inference.py` 中 `infer_one_cloud_trt` 的 `min_n=1024`
- [ ] 两个脚本运行结果与修复前一致
