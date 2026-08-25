# C++/Python 体素化管线对齐修复计划（voxelize-cpp-python-alignment）

## TL;DR

> **Quick Summary**: 修复 HPENet_v2 部署推理中 C++（deploy/CPP_trt1）与 Python（trt_inference.py）体素化管线产生的子云索引不一致问题——根因是 voxel_size 配置差 3 倍、RNG 种子被注释、排序稳定性不同。目标：**让 C++ 完全复现 Python 部署管线**，端到端 logits 一致（MAE < 1e-5）。
>
> **Deliverables**:
> - Python 侧：恢复 `np.random.seed(100)`、`argsort(kind='stable')`
> - C++ 侧：voxel_size 对齐（0.1f → 部署实际值 0.3）、test_start 对齐（0.0 → 0.2）
> - 验证链：golden 数据分步对照（哈希→排序→子云→端到端 logits）
> - 证据：`.omo/evidence/task-{N}-*.{ext}`
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: 任务1(诊断) → 任务2/3/4/5 → 任务6/7 → 任务8 → 任务9 → F1-F4 → 用户确认

---

## Context

### Original Request
体素化实现差异（C++ vs Python 的 voxelize 使用了不同的哈希/RNG，导致子云索引不同）：体素化产出的子云索引不同 → 不同的预处理输入 → 不同的 logits。

### Interview Summary
**Key Discussions**:
- 对比场景：C++（deploy/CPP_trt1）vs Python 部署推理（trt_inference.py / onnx_inference.py 对同一 PLY 的输出）
- 修复目标（用户明确选择）：**让 C++ 完全复现 Python（部署管线为准）**
- voxel_size：训练确为 0.3（此差异不属 C++/Python 对比范畴，列为附加项）
- 范围：完整修复计划（诊断 → 修复 → 端到端验证）
- 4 个后续问题用户未回答 → 按合理默认裁决（见下方"默认方案"）

**Research Findings**:
- C++ 侧真正的 voxelize 在 `deploy/` 下 5 个工程，分 GPU 混合版（CPP_trt/CPP_trt1/CPP_trt2：GPU FNV-1 + thrust::sort_by_key + NumpyMT19937 rng(seed=100)）与纯 CPU 版（CPP_onnx/CPP_onnx1：std::stable_sort + std::mt19937+std::shuffle，与 numpy 不兼容）
- Python 侧唯一核心实现：`openpoints/dataset/data_util.py::voxelize`（FNV-1a 整列乘+异或 → argsort → np.unique）+ `deploy/common.py::preprocess_test`（np.random.shuffle）
- 现成测试基座：`deploy/CPP_trt/scripts/gen_golden_data.py`（seed=42 RandomState）——已把 Python voxelize(mode=1)+fnv 输出存为 C++ 对照基准
- 测试基础设施：**无 pytest/CI**（AGENTS.md），验证靠对照实验脚本

### Metis Review
**Identified Gaps** (addressed):
- ⚠️ **部署 voxel_size 矛盾**：trt_inference.py 默认 cfgPath=hpenet-ll.yaml（radius: 0.3 #0.1 实际生效 0.3）≠ C++ main.cpp 默认 0.1f → 体素差 3 倍，是最大差异源 → 任务1 加诊断实证，任务4 对齐
- pipeline.cpp:491 `test_start = n_total * 0.0f` vs Python `all_files[n_total*0.2:]` → 文件集不同 → 任务5 对齐 0.2
- stats 文件不一致风险：C++ 默认 `stats_feat5.json`（main.cpp:28）vs Python `feat_stats_area5.pth`（trt_inference.py:165）→ 任务8 对照验证
- 确定性困境：Python seed(100) 被注释 → 无确定性基准，"完全复现"先恢复 seed → 任务2
- process_pointcloud（L73）/process_file（L273）重复 160 行 → 修改时两端同步（任务3/4/5 guardrail）

## Work Objectives

### Core Objective
让 C++ (CPP_trt1) 完全复现 Python 部署推理管线的 voxelize 结果：同一 PLY 输入 → 相同子云索引 → 相同预处理器输入 → 相同 logits（逐点 MAE < 1e-5）。

### Concrete Deliverables
- `deploy/trt_inference.py` / `deploy/onnx_inference.py`：恢复 `np.random.seed(100)`
- `openpoints/dataset/data_util.py` / `deploy/common.py`：`argsort(kind='stable')`
- `deploy/CPP_trt1/main.cpp`：voxel_size 默认值对齐部署实际值
- `deploy/CPP_trt1/src/pipeline.cpp`：test_start 0.0f → 0.2f
- 验证证据：`.omo/evidence/`

### Definition of Done
- [ ] 同一 PLY：C++ 与 Python 子云索引逐体素一致（set 成员相等）
- [ ] 同一 PLY：C++ 与 Python 端到端 logits 逐点 MAE < 1e-5
- [ ] C++ 连续 3 次运行输出完全一致（确定性）
- [ ] 证据文件存在于 `.omo/evidence/`

### Must Have
- 恢复 `np.random.seed(100)`（确定性是"完全复现"的前提）
- 统一稳定排序：Python `argsort(kind='stable')`，C++ 保持 thrust 稳定排序
- voxel_size 对齐部署实际生效值（任务1 实证确认）
- test_start 对齐（0.0 → 0.2）
- 分步对照验证（哈希 → 排序 → 子云 → logits），每步用 golden 数据

### Must NOT Have (Guardrails)
- **不得**修改 cpp 哈希常量（FNV_OFFSET/FNV_PRIME）——已实验验证与 Python 一致（MATCH）
- **不得**替换 NumpyMT19937 为 std::mt19937——前者已验证 numpy 兼容
- **不得**改动模型权重、onnx/engine 文件——修复仅在预处理层
- **不得**把 CPP_onnx*（std::mt19937+shuffle，与 numpy 不兼容）纳入主修复范围——用户焦点是 CPP_trt1，可作为后续可选任务
- **训练 voxel_size=0.3 与部署的差异**分析列为附加项，不阻塞本计划
- 不引入新的随机性来源；所有 RNG 固定种子
- 禁止删除/改动 `examples/*/main.py` 的 `import __init__` 路径 hack（AGENTS.md）

### 默认方案（用户未回答 4 问题，Prometheus 应用合理默认，覆盖可改）
1. 部署 voxel_size 默认 = hpenet-ll.yaml 实际生效 0.3（任务1 实证，若实证为 0.1 则 C++ 保持 0.1）
2. seed(100) 注释 = 失误 → 恢复
3. 排序 = 统一稳定排序
4. test_start = 纳入修复范围

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO（无 pytest/CI，仅 1 个 unittest 文件）
- **Automated tests**: None（对照实验脚本 + agent QA）
- **Framework**: 无（使用黄金数据对照脚本 + 端到端 logits 对比）

### QA Policy
每个任务 MUST 含 agent 执行的 QA 场景，证据存 `.omo/evidence/`。
- **库/算法对照**：Bash 运行 Python 脚本 + C++ 可执行程序，比对输出（逐位/数值容差）
- **CLI**：Bash 运行 trt_inference.py / C++ 可执行，断言退出码 + 输出文件
- **确定性**：同一命令运行 3 次，对比输出哈希

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (立即开始 - 诊断 + 基础修复，MAX PARALLEL):
├── Task 1: 诊断部署 voxel_size 实际生效值 [quick]
├── Task 2: Python 恢复 seed(100) [quick]
├── Task 3: Python 统一稳定排序 [quick]
├── Task 4: C++ voxel_size 对齐 [quick]
└── Task 5: C++ test_start 对齐 [quick]

Wave 2 (Wave 1 后 - golden 基座 + RNG 验证):
├── Task 6: 更新 gen_golden_data.py 生成稳定排序 golden 数据 [quick]
├── Task 7: C++ RNG(NumpyMT19937) vs numpy RandomState 一致性验证 [unspecified-high]
└── Task 8: stats 文件一致性对照 [quick]

Wave 3 (Wave 2 后 - 端到端验证):
├── Task 9: 端到端 C++ vs Python logits 一致性验证 [unspecified-high]
└── Task 10: 确定性复现验证 (3 次运行) + 批量多文件测试 [unspecified-high]

Wave FINAL (全部任务后 — 4 并行审查，然后用户确认):
├── Task F1: Plan Compliance Audit (oracle)
├── Task F2: Code Quality Review (unspecified-high)
├── Task F3: Real Manual QA (unspecified-high)
└── Task F4: Scope Fidelity Check (deep)
-> 呈现结果 -> 获取用户明确 ok

Critical Path: Task 1 → Task 4 → Task 6 → Task 9 → Task 10 → F1-F4 → 用户确认
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 1)
```

### Dependency Matrix
- **1**: - → 4
- **2**: - → 6, 9
- **3**: - → 6, 9
- **4**: 1 → 9
- **5**: - → 9, 10
- **6**: 2, 3 → 9
- **7**: 2 → 9
- **8**: - → 9
- **9**: 4, 5, 6, 7, 8 → 10
- **10**: 9 → F1-F4

### Agent Dispatch Summary
- **Wave 1**: T1-T5 → `quick`
- **Wave 2**: T6 → `quick`, T7 → `unspecified-high`, T8 → `quick`
- **Wave 3**: T9 → `unspecified-high`, T10 → `unspecified-high`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. 诊断部署 voxel_size 实际生效值

  **What to do**:
  - 实证确认 Python 部署管线运行时实际使用的 voxel_size：读取 `deploy/trt_inference.py` 的 `--cfgPath` 默认值（line 158-159），加载该 YAML 配置，打印 `cfg.model.encoder_args.radius` 实际生效值
  - 检查 `deploy/common.py::preprocess_test` 收到 voxel_size 参数的传值链（onnx_inference.py / trt_inference.py → common.py）
  - 结论落档：确认部署 voxel_size 是 0.3（hpenet-ll.yaml）还是 0.1（hpenet-s/b.yaml 或手动覆盖），写入 `.omo/evidence/task-1-voxel-size-diag.txt`
  - 同时列出 `deploy/CPP_trt1/src/main.cpp` 的 voxel_size 默认值来源（line 34：`float voxel_size = 0.1f;`）

  **Must NOT do**:
  - 不要修改任何代码（纯诊断任务）
  - 不要运行完整推理（只需配置读取与打印）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯读配置 + 打印，单文件小改动（或零改动）
  - **Skills**: []
    - 无需外部技能；涉及 numpy/yaml 基础读取

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Task 4 (C++ voxel_size 对齐依赖实证值)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `deploy/trt_inference.py:158-159` - `--cfgPath` 默认值 `cfgs/radar/hpenet-ll.yaml`
  - `cfgs/radar/hpenet-ll.yaml:17` - `radius: 0.3 #0.1`（0.1 被注释，实际生效 0.3）
  - `deploy/common.py:53-74` - `preprocess_test()` 接收 voxel_size
  - `deploy/CPP_trt1/src/main.cpp:34` - voxel_size 默认值 0.1f
  - `cfgs/radar/hpenet-s.yaml:17`, `cfgs/radar/hpenet-b.yaml:17` - radius: 0.1（对照组）
  - **WHY**: 这些引用共同回答"部署实际 voxel_size 是多少"——决定后续对齐方向（0.3 或 0.1）

  **Acceptance Criteria**:
  - [ ] `.omo/evidence/task-1-voxel-size-diag.txt` 存在，明确写出：cfgPath、radius 实际生效值、C++ main.cpp 默认值、结论（对齐目标值）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 诊断输出部署 voxel_size
    Tool: Bash
    Preconditions: conda 环境可运行 python（numpy/yaml 可用）
    Steps:
      1. 运行诊断脚本（读取 trt_inference.py 默认 cfgPath → 加载 YAML → 打印 radius）
      2. 断言输出包含 "radius = 0.3" 或明确的非 0.1 值（若实测为 0.1 则记录原因：手动覆盖/不同 cfg）
      3. 断言 C++ main.cpp 默认值被打印（0.1f）
    Expected Result: 诊断文件含 cfgPath + radius 实际值 + C++ 默认值 + 对齐目标结论
    Failure Indicators: 脚本无法加载 YAML（需检查 cfgPath 解析）或 radius 读取失败
    Evidence: .omo/evidence/task-1-voxel-size-diag.txt

  Scenario: [edge case] cfgPath 被 CLI 覆盖（非默认 hpenet-ll.yaml）
    Tool: Bash
    Preconditions: 诊断脚本支持 --cfgPath 参数
    Steps:
      1. 运行诊断脚本 --cfgPath cfgs/radar/hpenet-s.yaml
      2. 断言输出 radius = 0.1（覆盖生效）
      3. 记录两种配置下的结论差异（默认 0.3 vs 覆盖 0.1）
    Expected Result: 诊断文件同时记录默认与覆盖两种情形的 voxel_size 结论
    Failure Indicators: 覆盖参数不生效，输出仍是 0.3
    Evidence: .omo/evidence/task-1-voxel-size-diag.txt
  ```

  **Commit**: NO（纯诊断，无代码改动）

- [ ] 2. Python 部署管线恢复确定性种子 (np.random.seed(100))

  **What to do**:
  - 在 `deploy/trt_inference.py` 和 `deploy/onnx_inference.py` 中恢复被注释的 `np.random.seed(100)`（trt_inference.py:228、onnx_inference.py:246）
  - 确保 seed 设置在 `preprocess_test`/`np.random.shuffle` 被调用**之前**执行（检查调用顺序：seed 应在数据加载/预处理循环开始前）
  - 若无相关位置，则在 `main` 入口最前面（或 load_cfg 之后、处理循环之前）显式调用 `np.random.seed(100)`
  - 验证修改后同一 PLY 两次运行输出一致（确定性）

  **Must NOT do**:
  - 不要修改 `deploy/common.py` 中的 `preprocess_test` 逻辑（**必须**保持预处理语义不变，只依赖全局 seed）
  - 不要删除/改动其他 RNG 调用点（dataset/build.py:41 worker_init_fn 等）——本任务只处理部署脚本
  - 不要改用 `np.random.RandomState` 局部实例（除非无法用全局 seed 达成，且需与 C++ 侧 NumpyMT19937(seed=100) 语义核对）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 2 个单行变更 + 顺序检查
  - **Skills**: []
    - 纯 Python 标准库 + numpy

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: Task 6 (golden 数据生成需确定性), Task 9 (端到端对比)
  - **Blocked By**: None

  **References**:
  - `deploy/trt_inference.py:228` - `# np.random.seed(100)` 被注释处
  - `deploy/onnx_inference.py:246` - 同上
  - `deploy/common.py:53-74` - `preprocess_test()` 内 `np.random.shuffle(idx_part)`（L67）依赖全局 RNG 状态
  - `deploy/CPP_trt1/src/random_util.cpp` + `include/random_util.h` - NumpyMT19937 rk_seed(100) 参考实现（seed=100 来源）
  - `openpoints/utils/random.py` - 项目 set_random_seed 惯例（若需参考）
  - **WHY**: 恢复 seed 是确定性基准的前提；C++ 侧用 seed=100，Python 必须对齐同一种子

  **Acceptance Criteria**:
  - [ ] `deploy/trt_inference.py` 和 `deploy/onnx_inference.py` 中 seed 行已恢复为有效代码（非注释）
  - [ ] 修改后：两次运行同一单文件 PLY 子云结果一致（与原"两次运行不一致"实验形成对比）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 恢复 seed 后 Python 确定性验证
    Tool: Bash
    Preconditions: conda 环境 + 至少 1 个真实 PLY 文件路径（如 data/RadarClassi/radarfullwl/raw/ 下任意文件）
    Steps:
      1. 用伪造/最小输入调用 preprocess_test（或直接跑 trt_inference.py --num_files=1）第一次运行，保存子云索引/随机 key 输出到 /tmp/run1.pkl
      2. 第二次运行，保存到 /tmp/run2.pkl
      3. 断言 np.array_equal 或逐文件 sha256 一致
    Expected Result: run1 == run2（确定性成立）
    Failure Indicators: 两次结果仍不同 → seed 调用位置仍在 shuffle 之后，需移到更早位置
    Evidence: .omo/evidence/task-2-determinism.txt（记录两次 sha256）

  Scenario: [edge case] seed 调用位置晚于 shuffle（若失败可定位）
    Tool: Bash
    Preconditions: 已恢复 seed 行
    Steps:
      1. 人为把 seed(100) 移到 preprocess_test 内部 shuffle 之后（临时验证用，验证后恢复）
      2. 两次运行，断言结果不同
      3. 恢复正确位置（main 入口、shuffle 之前），断言两次运行一致
    Expected Result: 位置错误 → 不确定；位置正确 → 确定。证明调用顺序是关键
    Failure Indicators: 位置正确后仍不确定 → seed 值或调用点仍有遗漏
    Evidence: .omo/evidence/task-2-determinism.txt
  ```

  **Commit**: YES
  - Message: `fix(deploy): 恢复 np.random.seed(100) 确保部署确定性`
  - Files: `deploy/trt_inference.py`, `deploy/onnx_inference.py`
  - Pre-commit: `python -m py_compile deploy/trt_inference.py deploy/onnx_inference.py`

- [ ] 3. Python 体素化统一稳定排序 (argsort kind='stable')

  **What to do**:
  - 在 `openpoints/dataset/data_util.py::voxelize` 中把 `idx_sort = np.argsort(key)` 改为 `idx_sort = np.argsort(key, kind='stable')`
  - 检查 `deploy/common.py::preprocess_test`（L53-74）是否有独立的 argsort 调用（若有，同步改为 kind='stable'；若无则仅依赖 data_util.voxelize）
  - 检查 `openpoints/dataset/radar/s3disRadar.py` / 其他 voxelize 调用点是否有独立 argsort 需同步（遍历 `np.argsort` 引用）
  - **关键**：确认同一体素内的点序变为"原始点序"后，训练管线是否有依赖（训练用 mode=0，不受影响；但需确认 mode=1 的测试路径行为）

  **Must NOT do**:
  - 不要修改 `fnv_hash_vec` / `ravel_hash_vec` 哈希逻辑（已与 C++ 验证一致）
  - 不要修改 voxelize 的 mode=0 训练路径（每体素随机选点逻辑不动）
  - 不要改动 `np.unique` / `np.floor` 除法路径

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 1-2 处单参数变更 + 引用核对
  - **Skills**: []
    - 涉及 numpy argsort API 基础

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: Task 6 (golden 数据生成需与新排序对齐), Task 9 (端到端)
  - **Blocked By**: None

  **References**:
  - `openpoints/dataset/data_util.py:127-143` - `voxelize()` 主函数，`idx_sort = np.argsort(key)` L134
  - `deploy/common.py:53-74` - `preprocess_test()` 部署侧 voxelize 调用
  - `examples/segmentation/main.py:116-144` - 测试路径 multi_voxel 逻辑（若此处有 argsort 需核对）
  - `openpoints/dataset/radar/s3disRadar.py:89` - 训练侧 `voxelize(coord, voxel_size)` mode=0 调用
  - **WHY**: argsort 稳定性决定同体素内点序 → 决定 i%count 选出的代表点 → 决定子云成员；两侧必须一致

  **Acceptance Criteria**:
  - [ ] `data_util.py` 中 argsort 已带 `kind='stable'`
  - [ ] grep 确认无其他未同步的 `np.argsort(` 调用点（data_util/common.py/main.py 相关路径）
  - [ ] 对同一测试输入，返回值与预期稳定排序一致

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: stable 排序生效验证
    Tool: Bash (python)
    Preconditions: conda 环境
    Steps:
      1. 构造含重复 key 的小输入：coord = [[0,0,0],[0,0,0],[1,0,0]], voxel_size=1
      2. 调用 data_util.voxelize(coord, 1, mode=1)
      3. 断言 idx_sort 中 key 相等的两个同体素点的相对顺序 = 原输入顺序（稳定）
      4. 对比稳定/不稳定输出差异
    Expected Result: 同 key 点按原始出现顺序排列
    Failure Indicators: 排序不稳定或包装错误
    Evidence: .omo/evidence/task-3-stable-sort.txt

  Scenario: [edge case] 空输入 / 单点输入不崩溃
    Tool: Bash (python)
    Preconditions: conda 环境
    Steps:
      1. 调用 data_util.voxelize(np.empty((0,3)), 0.3, mode=1) → 断言返回空结果不抛异常
      2. 调用 data_util.voxelize(np.array([[0,0,0]]), 0.3, mode=1) → 断言单点结果 idx_sort 长度 1
    Expected Result: 空/单点输入均正常返回（与 C++ 侧边界行为对齐）
    Failure Indicators: 任一输入抛异常或返回错误形状
    Evidence: .omo/evidence/task-3-stable-sort.txt
  ```

  **Commit**: YES
  - Message: `fix(dataset): voxelize argsort 改为稳定排序(kind='stable')`
  - Files: `openpoints/dataset/data_util.py`, `deploy/common.py`
  - Pre-commit: `python -m py_compile openpoints/dataset/data_util.py deploy/common.py`

- [ ] 4. C++ voxel_size 对齐部署实际值

  **What to do**:
  - 根据任务 1 的实证结果，将 `deploy/CPP_trt1/src/main.cpp` 的 voxel_size 默认值（当前 0.1f，line 34）改为与 Python 部署实际生效值一致（预期 0.3f，若任务 1 证实为 0.1 则无需改动并记录）
  - 检查 voxel_size 是否支持命令行参数/配置读取（main.cpp 已有 `--voxel_size` CLI 参数，line 99/118——确认默认值来源与覆盖路径）
  - 若存在多个 CPP_trt* 工程（CPP_trt/CPP_trt1/CPP_trt2）共享默认值，仅修改用户聚焦的 CPP_trt1（guardrail：其他工程不动）
  - 同步检查 `deploy/common.py` 或 Python 侧是否有硬编码 0.1 需对账

  **Must NOT do**:
  - 不要修改 CPP_trt / CPP_trt2 / CPP_onnx* 工程（用户焦点是 CPP_trt1）
  - 不要修改模型 config（cfgs/radar/*.yaml 中的 radius 是模型配置，不是本任务对象——除非任务 1 证实配置本身错误）
  - 不要改动哈希/排序/RNG 逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单值变更 + 参数解析检查
  - **Skills**: []
    - C++ 基础（arg parsing）

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Task 9 (端到端对比需 voxel_size 一致)
  - **Blocked By**: Task 1 (实证结果)

  **References**:
  - `deploy/CPP_trt1/src/main.cpp:34` - voxel_size 默认值 0.1f
  - `deploy/CPP_trt1/src/main.cpp:99,118` - `--voxel_size` CLI 覆盖参数
  - `deploy/CPP_trt1/src/pipeline.cpp` - process_pointcloud 使用 voxel_size 处（L44,48,105,300）
  - `cfgs/radar/hpenet-ll.yaml:17` - radius: 0.3（任务 1 实证的依据）
  - `deploy/CPP_trt1/src/voxelizer.cu` - voxelize 内核（voxel_size 传入处）
  - **WHY**: voxel_size 差 3 倍导致体素分组完全不同，是最大差异源；对齐是端到端一致的前提

  **Acceptance Criteria**:
  - [ ] `main.cpp` 默认 voxel_size 已改为任务 1 实证值（或记录无需改动）
  - [ ] 确认 `--voxel_size` CLI 参数可覆盖默认值（line 99/118 已支持则确认即可）
  - [ ] 编译通过（若环境可行；否则语法检查）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: voxel_size 对齐验证
    Tool: Bash
    Preconditions: 任务 1 已完成（.omo/evidence/task-1-voxel-size-diag.txt 存在）
    Steps:
      1. grep main.cpp voxel_size 默认值
      2. 断言与 task-1 诊断文件中的"对齐目标值"一致
      3. 若可编译：cmake build 确认无报错
    Expected Result: 默认值 = 部署实际值（0.3 或诊断结论）
    Failure Indicators: 默认值仍为 0.1 且诊断结论为 0.3
    Evidence: .omo/evidence/task-4-voxel-size-aligned.txt

  Scenario: [edge case] CLI 覆盖 voxel_size 后仍可对齐
    Tool: Bash
    Preconditions: main.cpp 已对齐默认值，--voxel_size 参数存在
    Steps:
      1. 运行 C++ 程序 --voxel_size 0.1（显式覆盖）
      2. 确认运行不报错且 voxel_size 生效（日志/输出）
      3. 运行 C++ 程序（无参数）→ 默认值 = 部署值
    Expected Result: 显式覆盖与默认两条路径都可用
    Failure Indicators: 覆盖参数解析失败或默认值未生效
    Evidence: .omo/evidence/task-4-voxel-size-aligned.txt
  ```

  **Commit**: YES
  - Message: `fix(cpp): voxel_size 默认值对齐部署实际值`
  - Files: `deploy/CPP_trt1/src/main.cpp`
  - Pre-commit: 编译检查（若可行）

- [ ] 5. C++ test_start 对齐 Python 文件划分

  **What to do**:
  - 将 `deploy/CPP_trt1/src/pipeline.cpp:491` 的 `const int test_start = static_cast<int>(n_total * 0.0f);` 改为与 Python 一致的 `n_total * 0.2f`
  - 核对 Python 侧划分：`trt_inference.py:231` `all_files[int(n_total * 0.2):]`（后 17% 文件）——确认 C++ 使用相同公式
  - 检查 process_pointcloud（L73）与 process_file（L273）两处是否有重复的 test_start 逻辑需同步（Metis 发现约 160 行重复）
  - 确认文件排序一致：C++ `std::sort(ply_paths)` vs Python `sorted(...)`（通常一致，若文件名含特殊字符需复核）

  **Must NOT do**:
  - 不要重构/删除 process_pointcloud 或 process_file 的重复（超出本任务范围，仅同步 test_start 值）
  - 不要修改数据目录筛选逻辑的其他部分（glob 模式、扩展名过滤）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单值变更 + 一致性核对
  - **Skills**: []
    - C++ 基础

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Task 9, Task 10 (批量对比需同文件集)
  - **Blocked By**: None

  **References**:
  - `deploy/CPP_trt1/src/pipeline.cpp:491` - `test_start = n_total * 0.0f`
  - `deploy/trt_inference.py:231` - `all_files[int(n_total * 0.2):]`
  - `deploy/CPP_trt1/src/pipeline.cpp:73` 与 `:273` - 重复的 process 逻辑
  - **WHY**: C++ 处理全部文件而 Python 只处理后 17%，跨文件批量对比无意义；对齐划分使批量验证有效

  **Acceptance Criteria**:
  - [ ] `pipeline.cpp:491` 为 `0.2f`
  - [ ] 无其他残留 0.0f 的 test_start（grep 确认）
  - [ ] C++ 与 Python 文件集合一致（目录内有 N 个 PLY 时，C++ 处理后 17%）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: test_start 对齐验证
    Tool: Bash
    Preconditions: 目录下有已知数量的 PLY（如 n=50）
    Steps:
      1. grep pipeline.cpp test_start 行
      2. 断言为 0.2f（非 0.0f）
      3. 计算 C++ 应处理文件数 = 50 - int(50*0.2) = 40；Python 相同公式
    Expected Result: test_start 为 0.2f，两侧文件数公式一致
    Failure Indicators: 仍为 0.0f，或两侧公式不一致
    Evidence: .omo/evidence/task-5-test-start.txt

  Scenario: [edge case] 文件名排序一致性（C++ std::sort vs Python sorted）
    Tool: Bash
    Preconditions: 数据目录含编号文件名（如 frame_0001.ply ... frame_0050.ply）
    Steps:
      1. Python: sorted(glob('*.ply')) 前 5 个文件名
      2. C++: 日志/输出中的文件处理顺序前 5 个
      3. 断言前缀序列一致（编号文件名按字典序）
      4. 若目录含特殊字符/大小写混杂文件名：额外断言排序结果一致
    Expected Result: 两侧文件处理顺序一致
    Failure Indicators: 文件顺序不一致 → 批量对比的"同一文件"假设失效，需统一排序规则
    Evidence: .omo/evidence/task-5-test-start.txt
  ```

  **Commit**: YES
  - Message: `fix(cpp): test_start 对齐 Python 0.2 文件划分`
  - Files: `deploy/CPP_trt1/src/pipeline.cpp`
  - Pre-commit: 编译检查（若可行）

- [ ] 6. 更新 gen_golden_data.py 生成稳定排序 golden 数据

  **What to do**:
  - 阅读现有 `deploy/CPP_trt/scripts/gen_golden_data.py`（seed=42 RandomState 基座），理解其导出格式（Python voxelize(mode=1)+fnv 输出 → C++ 对照基准）
  - 更新该脚本（或创建副本 gen_golden_data_stable.py）：
    - 与任务 3 的 `argsort(kind='stable')` 对齐（若脚本内有自己的 argsort 实现需同步）
    - 输出追加/保持：discrete_coord、hash key、idx_sort、voxel_idx、count、子云成员（idx_points）
    - 保持 seed 可配置（默认沿用 42 或改为 100 对齐 C++ NumpyMT19937(seed=100)）
  - 生成 golden 数据文件（JSON/npy/binary），供任务 9 对比
  - 确认导出格式与 C++ 侧 `deploy/CPP_trt/scripts/` 或 CPP_trt1 测试的可读格式兼容（检查现有 test 代码如何消费 golden 文件）

  **Must NOT do**:
  - 不要修改 C++ 侧的 golden 消费代码（测试文件）
  - 不要改变 Python voxelize 核心实现（data_util.py）——只在 golden 生成脚本侧对齐
  - 不要修改 CPP_trt（非 CPP_trt1）工程代码——脚本只是基准生成器

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 脚本参数化 + 输出扩展，单文件小改动
  - **Skills**: []
    - numpy 序列化（np.save/json）

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖任务 3 的稳定排序）
  - **Parallel Group**: Wave 2 (with Tasks 7, 8)
  - **Blocks**: Task 9 (端到端对比需要 golden 基准)
  - **Blocked By**: Task 2, Task 3 (需确定性 + 稳定排序)

  **References**:
  - `deploy/CPP_trt/scripts/gen_golden_data.py` - 现有基准生成器（seed=42 RandomState）
  - `openpoints/dataset/data_util.py:127-143` - voxelize 实现（golden 数据源）
  - `deploy/CPP_trt1/src/voxelizer.cu` - C++ voxelize 输出结构（对齐导出字段）
  - **WHY**: golden 数据是"哈希→排序→子云"分步对照的基准；不生成则无法客观验证对齐

  **Acceptance Criteria**:
  - [ ] 脚本可运行，导出文件含 discrete_coord / key / idx_sort / voxel_idx / count / idx_points
  - [ ] golden 文件与任务 3 稳定排序语义一致（同输入输出可复现）
  - [ ] 至少 1 个真实 PLY（或最小合成输入）的 golden 数据已生成到 `.omo/evidence/golden/`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: golden 数据生成与稳定性
    Tool: Bash
    Preconditions: conda 环境 + 1 个 PLY 文件路径
    Steps:
      1. 运行 golden 生成脚本（--seed 100 --input <ply> --out .omo/evidence/golden/）
      2. 运行两次，断言输出文件 sha256 一致（确定性）
      3. 检查输出字段齐全（读 golden 文件头/JSON schema）
    Expected Result: 两次生成完全一致，字段完整
    Failure Indicators: 两次不一致（seed 未生效）或缺字段
    Evidence: .omo/evidence/task-6-golden-gen.txt

  Scenario: [edge case] 空/极小点云 golden 生成不崩溃
    Tool: Bash
    Preconditions: 可构造 0 点或 1 点的 PLY（或脚本支持合成输入）
    Steps:
      1. 运行 golden 脚本处理 1 点 PLY → 断言输出 idx_sort 长度 1、count 数组非空
      2. 若脚本支持 0 点：断言返回空结果不抛异常（与 C++ 侧边界行为对齐）
    Expected Result: 极小输入正常输出，无异常
    Failure Indicators: 空/单点输入抛异常或输出形状错误
    Evidence: .omo/evidence/task-6-golden-gen.txt
  ```

  **Commit**: YES
  - Message: `test(cpp): 更新 gen_golden_data.py 生成稳定排序 golden 数据`
  - Files: `deploy/CPP_trt/scripts/gen_golden_data.py`
  - Pre-commit: 脚本语法检查 + 试运行

- [ ] 7. C++ RNG (NumpyMT19937) vs numpy RandomState 一致性验证

  **What to do**:
  - 编写/复用验证脚本：对比 `deploy/CPP_trt1` 的 `NumpyMT19937 rng(seed=100)` 的 Fisher-Yates shuffle 输出与 Python `np.random.RandomState(100)` 的 shuffle 输出
  - 参考 `deploy/CPP_trt1/include/random_util.h:13-16` 注释中的参考值（seed=100 前 20 个 uniform_int(0,9999)）
  - 验证维度：
    a. uniform_int 序列一致性（RK 区间随机数与 numpy randint 对比）
    b. Fisher-Yates shuffle 顺序一致性（同数组 shuffle 后顺序）
    c. 关键：每文件独立 rng(seed=100) 的语义 vs Python 全局连续状态——确认 seed 模型对齐后的 shuffle 结果一致
  - 记录验证结果到 `.omo/evidence/task-7-rng-consistency.txt`

  **Must NOT do**:
  - 不要修改 C++ RNG 实现（已验证 numpy 兼容，本任务只验证）
  - 不要修改 Python 部署脚本的 seed 调用（任务 2 已处理）
  - 若发现不一致：记录差异并停止（不要静默修改实现）——报告给用户决策

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 跨语言 RNG 序列对比，需严谨的序列级验证（数值语义核对）
  - **Skills**: []
    - C++ 编译运行（若可行）+ numpy

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8)
  - **Blocks**: Task 9 (端到端需 RNG 一致)
  - **Blocked By**: Task 2 (Python seed 恢复)

  **References**:
  - `deploy/CPP_trt1/src/random_util.cpp` + `include/random_util.h` - NumpyMT19937 rk_seed/rk_interval/Fisher-Yates 实现 + 参考值注释
  - `deploy/CPP_trt1/src/voxelizer.cu` - `NumpyMT19937 rng(seed)` 使用处（每文件新建）
  - `deploy/common.py:53-74` - Python 侧 np.random.shuffle 使用处
  - `openpoints/utils/random.py` - 项目 set_random_seed 惯例
  - **WHY**: C++ 注释声称与 numpy 兼容但未经验证；这是"完全复现"的最后一环，必须序列级证明

  **Acceptance Criteria**:
  - [ ] 验证脚本可运行，输出 NumpyMT19937 vs RandomState(100) 对比结论
  - [ ] uniform_int 序列、shuffle 顺序 两者 MATCH 或差异明确列出

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: RNG 序列一致性验证
    Tool: Bash
    Preconditions: 可编译/运行 C++（或已有可执行文件）
    Steps:
      1. 运行 C++ RNG 测试输出（前 20 个 uniform_int(0,9999) + 一次 shuffle 结果）
      2. Python 侧 RandomState(100) 生成相同序列 + shuffle
      3. 逐项断言相等
    Expected Result: 序列完全一致
    Failure Indicators: 任何一项不匹配 → 记录并上报，不自行修改实现
    Evidence: .omo/evidence/task-7-rng-consistency.txt

  Scenario: [edge case] RNG 不匹配时的归因路径（不静默修复）
    Tool: Bash
    Preconditions: 存在任一不匹配项
    Steps:
      1. 打印第一个不匹配的随机数索引与两侧值
      2. 检查是否为"每文件独立 rng vs 全局连续状态"的 seed 模型差异（非算法差异）
      3. 若为接口序列差异：记录到证据文件，标注待用户决策（不改代码）
      4. 若为算法实现差异：记录并停止（C++ RNG 实现是已验证组件，默认不该改）
    Expected Result: 差异被准确分类（seed 模型 vs 算法），上报而非掩埋
    Failure Indicators: 差异未分类即被修改实现
    Evidence: .omo/evidence/task-7-rng-consistency.txt
  ```

  **Commit**: YES
  - Message: `test(cpp): 验证 NumpyMT19937 与 numpy RandomState 一致性`
  - Files: 验证脚本（.omo/scripts/ 或 deploy/CPP_trt1/tests/ 下）
  - Pre-commit: 脚本运行

- [ ] 8. stats 文件一致性对照

  **What to do**:
  - 对照 C++ 默认 `stats_feat5.json`（`deploy/CPP_trt1/src/main.cpp:28`）与 Python 使用的 `feat_stats_area5.pth`（trt_inference.py:165）中的特征归一化统计（mean/std 或 min/max）
  - 读取两者数值并逐一对比：若字段语义一致（同特征同统计量），断言数值 MATCH
  - 若字段名/语义不同，记录差异并判断是否影响特征归一化结果（若 C++ 部署实际使用与 Python 不同统计 → 归一化输入不同 → logits 不同）
  - 结论写入 `.omo/evidence/task-8-stats-consistency.txt`；若发现实质差异（归一化输入会不同），标注为任务 9 的已知风险点，并建议修复方向（让 C++ 读取与 Python 相同的 stats 来源）

  **Must NOT do**:
  - 不要修改 stats 文件内容或 Python 读取逻辑
  - 不要修改 C++ stats 读取代码（除非任务 9 证实 logits 差异来自 stats，且用户批准后单独处理）
  - 不要假设文件格式——用实际读取验证（.pth 用 torch.load，.json 用 json.load）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 读取两个文件并对比数值
  - **Skills**: []
    - torch.load + json.load

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7)
  - **Blocks**: Task 9 (归一化输入一致性前提)
  - **Blocked By**: None

  **References**:
  - `deploy/CPP_trt1/src/main.cpp:28` - `stats_feat5.json` 路径
  - `deploy/trt_inference.py:165` - `feat_stats_area5.pth` 路径
  - `deploy/CPP_trt1/src/pipeline.cpp` - 特征归一化使用 stats 处
  - `deploy/common.py:88-115` - Python 侧 preprocess_subcloud 归一化
  - **WHY**: 即使体素化完全一致，归一化统计不同也会导致 logits 不同；必须先排除此差异源

  **Acceptance Criteria**:
  - [ ] 两个 stats 文件已读取，数值对比结果记录
  - [ ] 结论明确：MATCH（归一化一致）或 DIFF（列出差异字段，标为风险点）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: stats 数值对照
    Tool: Bash (python)
    Preconditions: conda 环境，两个文件路径可读
    Steps:
      1. 加载 stats_feat5.json（json）与 feat_stats_area5.pth（torch.load）
      2. 对齐特征顺序（按 feature_keys: x,heights 或文件内顺序）
      3. 逐特征断言 mean/std（或 min/max）数值相等（容差 1e-6）
    Expected Result: MATCH 或 DIFF 明确记录
    Failure Indicators: 无法加载任一方文件
    Evidence: .omo/evidence/task-8-stats-consistency.txt

  Scenario: [edge case] stats 语义不对齐（字段名/特征顺序不同）
    Tool: Bash (python)
    Preconditions: 两个文件均可加载
    Steps:
      1. 对比字段名集合：若完全一致 → 正常数值对比
      2. 若字段名不一致 → 按特征顺序（feature_keys: x,heights）对齐后重试
      3. 若特征语义不可对齐 → 记录 DIFF 结论与无法对齐的原因（feature_keys 或文件内顺序），标为任务 9 风险点
    Expected Result: 语义对齐后得出 MATCH/DIFF；或明确记录"不可对齐"结论
    Failure Indicators: 特征顺序假设错误导致误判 MATCH
    Evidence: .omo/evidence/task-8-stats-consistency.txt
  ```

  **Commit**: YES
  - Message: `test(deploy): stats 文件一致性对照`
  - Files: 验证脚本
  - Pre-commit: 脚本运行

- [ ] 9. 端到端 C++ vs Python logits 一致性验证

  **What to do**:
  - 对同一 PLY 文件：分别运行 Python 部署管线（trt_inference.py，含任务 2 恢复的 seed(100)）与 C++（CPP_trt1，含任务 4/5 对齐的 voxel_size/test_start）
  - 使用任务 6 的 golden 数据分步比对：
    a. 哈希层：Python 导出 discrete_coord+hash vs C++ fnv 输出逐位一致
    b. 排序层：idx_sort 逐位一致（任务 3 稳定排序后）
    c. 子云层：每轮子云 set(idx_part) 成员一致
    d. 端到端：每点 logits 对比，MaxAbsDiff < 1e-5
  - 若分步定位到差异，记录差异层（哈希/排序/RNG/voxel_size/stats）——这正是分步验证的价值
  - 输出结论到 `.omo/evidence/task-9-endtoend.txt`（含各层 PASS/FAIL + MAE 数值）
  - 注意：任务 8 若发现 stats DIFF，此处需在"stats 已知差异"的前提下解释 logits 差异的归属

  **Must NOT do**:
  - 不要修改模型/引擎/onnx（推理引擎保持原样）
  - 若发现差异：先归因（哪个修复未生效/哪个层还有差异），不要盲目改代码
  - 不要为了通过测试而放宽 MAE 阈值（1e-5 是硬标准）
  - 不要跳过哈希/排序/子云分步验证直接对比 logits（难定位）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 多步验证链 + 数值断言，需要严格逐层核验
  - **Skills**: []
    - Python + numpy + C++ 运行 + 数值对比

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖全部前置修复与 golden 数据）
  - **Parallel Group**: Wave 3 (with Task 10)
  - **Blocks**: Task 10 (批量验证)
  - **Blocked By**: Task 4, Task 5, Task 6, Task 7, Task 8 (全部前置)

  **References**:
  - `.omo/evidence/task-6-golden/*` - golden 基准数据
  - `deploy/trt_inference.py` - Python 部署入口（seed(100) 已恢复）
  - `deploy/CPP_trt1/` - C++ 部署工程（voxel_size/test_start 已对齐）
  - `deploy/CPP_trt/scripts/gen_golden_data.py` - golden 生成器（任务 6 更新）
  - `deploy/CPP_trt1/src/pipeline.cpp` - C++ logits 输出结构
  - **WHY**: 这是验证"完全复现"的核心任务——分层验证保证差异可归因，端到端 MAE 保证实际推理结果一致

  **Acceptance Criteria**:
  - [ ] 哈希层、排序层、子云层逐位/逐成员一致（PASS）
  - [ ] 端到端 logits MaxAbsDiff < 1e-5
  - [ ] 各层结果记录到 `.omo/evidence/task-9-endtoend.txt`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 单文件端到端 logits 一致性
    Tool: Bash
    Preconditions: 任务 2/3/4/5/6/7/8 已完成；1 个真实 PLY
    Steps:
      1. Python 侧运行 trt_inference.py --num_files=1，保存每点 logits 到 /tmp/py_logits.npy
      2. C++ 侧运行对应单文件，输出 logits 到 /tmp/cpp_logits.npy
      3. 运行对比脚本：逐点 MaxAbsDiff，断言 < 1e-5
      4. 分步：哈希 key 数组逐位比对（np.array_equal）；idx_sort 逐位比对；子云成员逐轮 set 相等
    Expected Result: 四层全部 PASS，MAE 数值记录
    Failure Indicators: 任一层 FAIL → 归因（voxel_size/RNG/排序/stats）并修复对应任务后重跑
    Evidence: .omo/evidence/task-9-endtoend.txt

  Scenario: [edge case] 分步定位失败层（不直接盲目改代码）
    Tool: Bash
    Preconditions: 端到端对比出现任一 FAIL
    Steps:
      1. 比较哈希数组：逐位比对，若 FAIL → 差异在 float32/64 除法边界（回任务 3/诊断）
      2. 若哈希 PASS、idx_sort FAIL → 排序差异（回任务 3）
      3. 若排序 PASS、子云成员 FAIL → RNG/seed 差异（回任务 2/7）
      4. 若子云 PASS、logits FAIL → stats 归一化差异（回任务 8）
    Expected Result: 明确归因到具体层，记录到任务 9 证据文件
    Failure Indicators: 无法定位 → 补充中间层导出（C++ 落盘中间结果）再对比
    Evidence: .omo/evidence/task-9-endtoend.txt
  ```

  **Commit**: YES
  - Message: `test(deploy): 端到端 logits 一致性验证`
  - Files: 验证脚本 + 证据
  - Pre-commit: 验证运行

- [ ] 10. 确定性复现 + 批量多文件验证

  **What to do**:
  - 确定性：同一命令运行 C++ pipeline 3 次（同一输入），断言 3 次输出完全一致（sha256 或逐字节）
  - 批量：在任务 5 对齐的文件集合上（后 17% 文件），运行 C++ 与 Python 各一批，逐文件断言 logits 一致（MAE < 1e-5）
  - 检查批量结果的异常文件（若有 FAIL 文件，归因并报告）
  - 输出 `.omo/evidence/task-10-batch.txt`（文件清单 + 每文件 MAE + 通过率）

  **Must NOT do**:
  - 不要缩小测试集来"碰巧通过"——必须用任务 5 定义的全部文件
  - 不要跳过确定性验证（C++ 3 次一致性是"完全复现"的稳定性前提）
  - 不要修改推理引擎（与任务 9 相同 guardrail）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 批量运行 + 逐文件数值断言 + 异常归因
  - **Skills**: []
    - Bash 脚本 + numpy + 日志分析

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖任务 9 的单文件验证通过）
  - **Parallel Group**: Wave 3 (with Task 9)
  - **Blocks**: F1-F4 (Final Wave)
  - **Blocked By**: Task 9

  **References**:
  - `deploy/CPP_trt1/src/pipeline.cpp:491` - test_start 0.2f（任务 5 已改）
  - `deploy/trt_inference.py:231` - Python 文件划分（同公式）
  - `deploy/trt_inference.py` 输出结构 - 每点 logits 保存方式
  - **WHY**: 批量验证证明修复对所有文件生效（非单文件巧合），确定性证明无隐藏随机性

  **Acceptance Criteria**:
  - [ ] C++ 3 次运行输出 sha256 完全一致
  - [ ] 全部（或 100% - 已归因异常）文件 logits MAE < 1e-5
  - [ ] `.omo/evidence/task-10-batch.txt` 记录文件清单 + 通过率

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 批量 + 确定性验证
    Tool: Bash
    Preconditions: 任务 9 单文件 PASS；目录含 N 个 PLY
    Steps:
      1. 运行 C++ pipeline（后 17% 文件）3 次，断言 3 份输出目录 sha256 一致
      2. 运行 Python trt_inference.py 同文件集
      3. 逐文件对比 logits，统计 PASS/FAIL 文件数与 MAE
    Expected Result: 确定性成立；通过率 100%（或异常文件已归因）
    Failure Indicators: 3 次输出不一致 → 隐藏随机源；FAIL 文件未归因
    Evidence: .omo/evidence/task-10-batch.txt

  Scenario: [edge case] 批量中出现 FAIL 文件的归因（不缩小测试集逃避）
    Tool: Bash
    Preconditions: 批量验证发现 ≥1 个 FAIL 文件
    Steps:
      1. 对比该文件的 C++/Python 输入（点数、坐标范围）与 PASS 文件差异
      2. 检查是否为边界点（float32/64 除法边界、负坐标环绕）→ 对应任务 3/9 归因
      3. 若为固定系统性差异 → 记录到证据文件并上报（不静默修复）
    Expected Result: FAIL 文件被归因（数据特征 or 修复遗漏），记录于 task-10-batch.txt
    Failure Indicators: FAIL 文件被从测试集剔除而未解释
    Evidence: .omo/evidence/task-10-batch.txt
  ```

  **Commit**: YES
  - Message: `test(deploy): 确定性复现 + 批量验证`
  - Files: 验证脚本 + 证据
  - Pre-commit: 验证运行

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**

- [ ] F1. **Plan Compliance Audit** — `oracle`
  读计划全文。逐条验证 Must Have 已实现（读文件、跑命令）；搜索 Must NOT Have 违禁模式（std::mt19937、改哈希常量、改权重），违者带 file:line 拒绝。检查 .omo/evidence/ 中证据文件齐全。对照交付物清单。
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  检查所有改动文件：死代码/注释残留（seed 是否恢复、kind='stable' 是否生效）、未用 import、AI slop（过度注释、泛化命名）。运行 Python 语法检查（python -m py_compile 改动文件）。确认 C++ 修改未破坏构建（如果可行，运行 cmake build）。
  Output: `Python Syntax [PASS/FAIL] | C++ Build [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  从干净状态执行每个任务的全部 QA 场景。重点：运行任务 9/10 端到端验证命令，确认 MAE < 1e-5、确定性成立。将证据存到 .omo/evidence/final-qa/。
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  逐任务读 "What to do" → 读实际 diff（git log/diff 或工作区 diff）。验证 1:1：计划内全部实现、无越界改动（尤其：未改 CPP_onnx*、未改模型权重、未改训练管线、未删 __init__ hack）。
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `fix(deploy): 诊断并确认部署 voxel_size 实际生效值` - 诊断脚本
- **2**: `fix(deploy): 恢复 np.random.seed(100) 确保确定性` - trt_inference.py, onnx_inference.py
- **3**: `fix(dataset): voxelize argsort 改为稳定排序(kind='stable')` - data_util.py, common.py
- **4**: `fix(cpp): voxel_size 默认值对齐部署实际值` - main.cpp
- **5**: `fix(cpp): test_start 对齐 Python 0.2 文件划分` - pipeline.cpp
- **6**: `test(cpp): 更新 gen_golden_data.py 生成稳定排序 golden 数据` - gen_golden_data.py
- **7**: `test(cpp): 验证 NumpyMT19937 与 numpy RandomState 一致性` - 验证脚本
- **8**: `test(deploy): stats 文件一致性对照` - 验证脚本
- **9**: `test(deploy): 端到端 logits 一致性验证` - 验证脚本
- **10**: `test(deploy): 确定性复现 + 批量验证` - 验证脚本

> 注：如需逐个 git commit，请用户明确允许后执行（AGENTS.md：禁止未经允许的 git 操作）。默认建议工作区留改，单次 commit 由用户决定。

---

## Success Criteria

### Verification Commands
```bash
# 任务9: 端到端 logits 对比（示例）
python deploy/trt_inference.py --num_files=1  # Expected: 输出并保存 logits
# C++ 侧运行对应单文件
# 对比脚本输出: MaxAbsDiff < 1e-5, PASS

# 任务10: 确定性（同一命令 3 次）
# Expected: 3 次输出文件 sha256 完全一致

# 任务7: RNG 一致性
python scripts/verify_rng.py  # Expected: NumpyMT19937 vs RandomState(100) shuffle 一致
```

### Final Checklist
- [ ] 所有 "Must Have" 已实现
- [ ] 所有 "Must NOT Have" 均未触碰
- [ ] 端到端 logits MAE < 1e-5
- [ ] 确定性验证通过（3 次运行一致）
- [ ] 用户明确批准