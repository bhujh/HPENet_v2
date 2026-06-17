# CPP_onnx 重构：process_directory 删除 + PLY 输出 + 批量推理保留

## TL;DR

> **Quick Summary**: 删除 `process_directory` 类方法，将批量推理 for-loop 移到 `main.cpp`；新增 `write_annotated_ply()` PLY 写入；保留准确率计算；CLI 改为 `--data_dir`（全量）+ `--output_dir`。
>
> **Deliverables**:
> - 修改 `onnx_inference.h` — 删除 `process_directory` 声明
> - 修改 `onnx_inference.cpp` — 删除 `process_directory` 实现、label 可选加载、新增 `write_annotated_ply()`
> - 修改 `main.cpp` — for-loop 批量推理 + PLY 输出 + 准确率汇总
> - 修改 `verify.py` — 适配新 CLI
>
> **Estimated Effort**: Quick
> **Parallel Execution**: NO — sequential

---

## Context

### Original Request
CPP_onnx 不需要 `process_directory` 函数，将批量处理逻辑直接写在 main.cpp 中。保留准确率计算。保留 data_dir 输入。每个文件额外输出标注 PLY。

### Interview Summary
**Key Decisions**:
- CLI: `--data_dir`（遍历全部 .ply）+ `--onnx` + `--stats_file` + `--output_dir`（标注 PLY 输出目录）
- 全部 .ply 文件处理，不用 83% 分割，不用 `--num_files`
- 输出 ASCII PLY，全部 7 字段保留，label 用推理结果覆盖
- 保留准确率：每文件单独打印 + 最终总体汇总
- 输入 PLY 无 label 字段不报错（填充 0）
- 静默覆盖已有输出文件

### Metis Review
**关键发现**:
- **CPP_trt 没有 PLY 写入参考** — PLY writer 需从零手写（基于 tinyply 的 `add_properties_to_element` + `write` API）
- **label 对比不在 process_file 中** — 在 main.cpp 第 36-41 行
- **load_data_ply 强制要求 7 个字段** — 需改为可选 label
- **verify.py 使用旧 CLI** — 必须同步更新

---

## Work Objectives

### Core Objective
重构 CPP_onnx：删除 `process_directory` 类方法，将批量处理逻辑移入 `main.cpp` 的 for-loop；新增每个文件的标注 PLY 输出；保留准确率计算。

### Concrete Deliverables
- `deploy/CPP_onnx/onnx_inference.h` — 删除 `process_directory` 声明
- `deploy/CPP_onnx/onnx_inference.cpp` — 删除 `process_directory`、label 可选、新增 `write_annotated_ply()`
- `deploy/CPP_onnx/main.cpp` — 新 CLI + for-loop 批量推理 + 准确率汇总
- `deploy/CPP_onnx/verify.py` — 适配新 CLI

### Definition of Done
- [x] `--data_dir <dir> --output_dir <dir>` 批量推理成功，每个文件输出标注 PLY
- [x] 输出 PLY 用 Python tinyply 可读取，字段正确
- [x] 准确率每文件打印 + 总体均值
- [x] `--help` 不显示 `--num_files`
- [x] `verify.py` 正常运行

### Must Have
- 批量目录推理：遍历 `data_dir` 下全部 `.ply` 文件
- 每个文件输出标注 PLY 到 `output_dir`
- 准确率计算（每文件 + 总体汇总）
- CLI: `--data_dir`, `--onnx`, `--stats_file`, `--output_dir`
- 输出 ASCII PLY，保留全部 7 个字段

### Must NOT Have (Guardrails)
- **NO process_directory 类方法** — 彻底删除
- **NO --num_files / --ply 参数** — 完全移除
- **NO 83% test split** — 全部文件
- **NO PLY 格式选项** — 仅 ASCII
- **NO 新依赖** — 只用 tinyply
- **NO 推理逻辑改动** — voxelize/preprocess/ONNX/scatter 不动
- **NO 不输出 PLY 时不写文件** — 必须每个文件都输出

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None
- **Verification**: Agent-executed QA via Bash

---

## Execution Strategy

### Sequential Execution

```
Task 1 → Task 2 → Task 3 → Task 4 → F1-F3
```

| 序号 | 任务 | 前置 | 产出 |
|------|------|------|------|
| 1 | 删除 process_directory + label 可选 | - | onnx_inference.h/cpp |
| 2 | 新增 PLY 写入 write_annotated_ply() | 1 | onnx_inference.cpp |
| 3 | main.cpp CLI 改造 | 1,2 | main.cpp |
| 4 | verify.py 适配 | 3 | verify.py |

---

## TODOs

- [x] 1. 删除 process_directory + label 可选加载

  **What to do**:
  - **onnx_inference.h**: 删除 `process_directory()` 声明
  - **onnx_inference.cpp**:
    - 删除 `process_directory()` 实现（约第 481-506 行）
    - 修改 `load_data_ply()`: 使 `label` 字段变为可选。当前强制要求 7 个字段。改为：先读前 6 个字段（x,y,z,rcs,snr,v），尝试读 label——若 PLY 中不存在则用 0 填充 `pc.label`。

  **Must NOT do**:
  - 不修改 `PointCloud` 结构体
  - 不修改推理核心逻辑

  **Recommended Agent Profile**: `quick`
  **Parallelization**: Sequential，无前置依赖

  **References**:
  - `deploy/CPP_onnx/onnx_inference.h:50-62` — process_directory 声明
  - `deploy/CPP_onnx/onnx_inference.cpp:100-110` — load_data_ply label 读取
  - `deploy/CPP_onnx/onnx_inference.cpp:481-506` — process_directory 实现

  **Acceptance Criteria**:
  - [ ] `process_directory` 在 .h 和 .cpp 中不存在
  - [ ] 无 label 字段的 PLY 不报错
  - [ ] 编译通过

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 编译 + process_directory 确认移除
    Tool: Bash
    Steps:
      1. cd deploy/CPP_onnx && cmake -B build && cmake --build build 2>&1
      2. Assert 退出码 0
      3. grep -r "process_directory" deploy/CPP_onnx/onnx_inference.h deploy/CPP_onnx/onnx_inference.cpp → 无结果
    Evidence: .omo/evidence/task-1-compile.txt
  ```

  **Commit**: YES (groups with Task 4)
  - Message: `refactor(onnx): delete process_directory, make label optional, add PLY output`

- [x] 2. 新增 PLY 写入函数 write_annotated_ply()

  **What to do**:
  - 在 `onnx_inference.cpp` 中新增自由函数：
    ```cpp
    void write_annotated_ply(const std::string& output_path,
                             const PointCloud& pc,
                             const std::vector<int>& predictions);
    ```
  - 使用 tinyply 写入 ASCII PLY：
    1. `std::filebuf` + `std::ostream` 打开 output_path
    2. `tinyply::PlyFile`，定义 vertex element: 7 属性 (x,y,z=float32, rcs,snr,v=float32, label=int32)
    3. `ply.add_properties_to_element(...)` + `ply.write(os, false)` (ASCII)
  - **注意**: `add_properties_to_element` 接受 `const uint8_t*` 非持有指针。所有 buffer 在函数内部作为局部 vector 分配，确保生命周期覆盖 write。
  - 添加 `#include <fstream>` 如缺失

  **Must NOT do**:
  - 不引入新库 — 只用 tinyply
  - 不修改 tinyply.h

  **Recommended Agent Profile**: `deep` — tinyply 写入 API 无现成参考，需自行理解

  **References**:
  - `deploy/CPP_trt/include/tinyply/tinyply.h` — tinyply 写入 API
  - `deploy/CPP_onnx/onnx_inference.cpp:85-135` — tinyply 读字段方式
  - `deploy/CPP_onnx/onnx_inference.h:12-19` — PointCloud 定义

  **Acceptance Criteria**:
  - [ ] 编译通过
  - [ ] 输出 PLY 可被 Python tinyply 正常读取
  - [ ] 7 个属性正确，label 为 int32 且值仅为 0/1
  - [ ] point count 与输入一致

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: PLY 写入 + Python 验证
    Tool: Bash
    Steps:
      1. write_annotated_ply("/tmp/test.ply", pc, predictions)
      2. python3 -c "
    import tinyply
    with open('/tmp/test.ply','rb') as f:
        ply=tinyply.PlyFile.read(f)
        v=ply['vertex']
        print('count:',v.count)
        print('props:',[p.name for p in v.properties])
        d=v.data
        print('label_set:',set(d['label']))
    "
      3. Assert count == pc.num_points
      4. Assert props == ['x','y','z','rcs','snr','v','label']
      5. Assert label_set subset of {0,1}
    Evidence: .omo/evidence/task-2-ply-write.txt
  ```

  **Commit**: YES (groups with Task 4)

- [x] 3. main.cpp CLI 改造

  **What to do**:
  - 完全替换 main.cpp：
    - 新 CLI: `--data_dir <dir>` / `--output_dir <dir>` / `--onnx <model>` / `--stats_file <file>`（默认 `stats.json`）
  - main 流程：
    1. 解析 4 个 CLI 参数，缺失打印 usage 退出
    2. 创建 `output_dir` 如果不存在（`mkdir -p` 类似逻辑，C++ 用 `std::filesystem::create_directories`）
    3. 扫描 `data_dir` 下所有 `.ply` 文件，排序
    4. for 每个 .ply 文件：
       a. `PointCloud pc = load_data_ply(filepath)`  — 获取原始数据
       b. `auto result = pipeline.process_file(filepath)` — 推理
       c. `float acc = (result.predictions == pc.label).mean()` — 计算准确率
       d. 生成输出路径：`output_dir / basename(filepath)`（同名）
       e. `write_annotated_ply(outpath, pc, result.predictions)` — 写入标注 PLY
       f. 打印 `[i/total] filename  acc=x.xxxx  latency=xxxms`
    5. 循环结束后打印总体均值：`avg_acc=x.xxxx`
  - usage 示例:
    ```
    Usage: hpenet_onnx_infer --data_dir <dir> --output_dir <dir> --onnx <model.onnx> [--stats_file stats.json]
    ```

  **Must NOT do**:
  - 不保留 `--num_files` / `--ply` 参数
  - 不做 83% test split
  - 不调用 `process_directory`（该函数已被删除）
  - 不在 main.cpp 中新增除 for-loop 外的复杂逻辑

  **Recommended Agent Profile**: `quick` — 参数替换 + for-loop

  **References**:
  - `deploy/CPP_onnx/main.cpp` — 当前代码
  - `deploy/CPP_onnx/onnx_inference.h:19-23` — InferenceResult 定义
  - `deploy/CPP_onnx/onnx_inference.h:56` — process_file 签名

  **Acceptance Criteria**:
  - [ ] `--data_dir <dir> --output_dir <dir>` 批量推理成功
  - [ ] 每个文件输出标注 PLY 到 output_dir
  - [ ] 准确率每文件打印 + 总体均值
  - [ ] `--help` 无 `--num_files` / `--ply`
  - [ ] 无 label 的文件准确率显示为 N/A

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 批量推理 + 准确率
    Tool: Bash
    Steps:
      1. rm -rf /tmp/cpp_onnx_out && LD_LIBRARY_PATH=./lib ./build/hpenet_onnx_infer \
         --data_dir ../../data/RadarClassi/radarfull/raw \
         --output_dir /tmp/cpp_onnx_out \
         --onnx ../../deploy/onnx_model.onnx \
         --stats_file stats.json
      2. Assert 退出码 0
      3. ls /tmp/cpp_onnx_out/*.ply | wc -l → 应该等于 data_dir 中 .ply 文件数
      4. Assert 输出包含每文件准确率
      5. Assert 最后一行包含 avg_acc
    Evidence: .omo/evidence/task-3-batch.txt

  Scenario: --help 正确
    Tool: Bash
    Steps:
      1. ./build/hpenet_onnx_infer --help
      2. Assert 包含 --data_dir, --output_dir, --onnx, --stats_file
      3. Assert 不包含 --num_files, --ply
    Evidence: .omo/evidence/task-3-help.txt

  Scenario: output_dir 自动创建
    Tool: Bash
    Steps:
      1. rmdir /tmp/cpp_onnx_auto 2>/dev/null;  # ensure doesn't exist
      2. ./build/hpenet_onnx_infer --data_dir <dir_with_1_ply> --output_dir /tmp/cpp_onnx_auto ...
      3. Assert /tmp/cpp_onnx_auto 目录被创建且有输出文件
    Evidence: .omo/evidence/task-3-mkdir.txt
  ```

  **Commit**: YES (groups with Task 4)

- [x] 4. verify.py 适配新 CLI

  **What to do**:
  - 更新 `verify.py` 适配新 CLI：
    - 旧: `--data_dir` / `--num_files` → 扫描目录选测试文件
    - 新: `--data_dir` / `--output_dir` / `--onnx` / `--stats_file` / `--cpp_binary`
  - 新流程：
    1. 运行 C++ 二进制: `./hpenet_onnx_infer --data_dir <dir> --output_dir <tmpdir> ...`
    2. 对每个输出 PLY，读取 label 字段作为 C++ 预测
    3. 运行 Python ONNX 推理得到 Python 预测
    4. 对比两者 label
    5. 输出逐文件差异 + 总体汇总
  - 保留原有 ONNX 模型加载和 Python 推理逻辑不变

  **Must NOT do**:
  - 不改变 Python ONNX 推理逻辑

  **Recommended Agent Profile**: `quick`

  **References**:
  - `deploy/CPP_onnx/verify.py` — 当前代码
  - `deploy/common.py` — load_data_ply Python 实现

  **Acceptance Criteria**:
  - [ ] `verify.py --data_dir <dir> --output_dir <tmpdir> ...` 正常运行
  - [ ] 输出 C++ vs Python 对比
  - [ ] 退出码 0

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: verify.py 端到端
    Tool: Bash
    Steps:
      1. cd deploy/CPP_onnx && python3 verify.py \
         --data_dir ../../data/RadarClassi/radarfull/raw \
         --output_dir /tmp/cpp_onnx_vout \
         --onnx ../../deploy/onnx_model.onnx \
         --cpp_binary ./build/hpenet_onnx_infer \
         --stats_file stats.json
      2. Assert 退出码 0
      3. Assert 输出对比结果
    Evidence: .omo/evidence/task-4-verify.txt
  ```

  **Commit**: YES (groups with Task 4)
  - Message: `refactor(onnx): delete process_directory, make label optional, add PLY output`

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — 编译 + Must Have/Must NOT Have 检查
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Build [PASS/FAIL] | VERDICT`

- [x] F2. **功能测试** — 批量推理 + 准确率正确 + PLY 输出可读
  Output: `Inference [N files] | Accuracy [correct] | PLY [valid N/N] | VERDICT`

- [x] F3. **verify.py 一致性** — Python vs C++ 预测对比
  Output: `Consistency [reasonable] | VERDICT`

---

## Commit Strategy

一次提交:
- Message: `refactor(onnx): delete process_directory, make label optional, add PLY output`
- Files: `deploy/CPP_onnx/onnx_inference.h`, `deploy/CPP_onnx/onnx_inference.cpp`, `deploy/CPP_onnx/main.cpp`, `deploy/CPP_onnx/verify.py`

---

## Success Criteria

### Verification Commands
```bash
# 编译
cd deploy/CPP_onnx && cmake -B build && cmake --build build

# 批量推理
LD_LIBRARY_PATH=./lib ./build/hpenet_onnx_infer \
  --data_dir ../../data/RadarClassi/radarfull/raw \
  --output_dir /tmp/cpp_onnx_out \
  --onnx ../../deploy/onnx_model.onnx \
  --stats_file stats.json

# 验证输出 PLY
python3 -c "import tinyply; ..." /tmp/cpp_onnx_out/0000260.ply

# verify.py
cd deploy/CPP_onnx && python3 verify.py --data_dir ... --output_dir ... --onnx ... --cpp_binary ... --stats_file stats.json
```

### Final Checklist
- [x] 编译通过
- [x] 批量推理正常运行
- [x] 每个文件有对应标注 PLY 输出
- [x] 准确率每文件打印 + 总体均值
- [x] process_directory 完全删除
- [x] verify.py 正常运行
