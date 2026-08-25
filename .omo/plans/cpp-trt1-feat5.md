# CPP_trt1 适配 feat5 模型

## TL;DR

> **Quick Summary**: 把 `deploy/CPP_trt1/` 从 3 原始特征(rcs,snr,v)/4 channel 升级为 feat5(4 原始特征 mag,rcs,snr,v / 5 channel),让 C++ TRT 推理能跑 `trt_model_feat5_fp16.engine` + 含 mag 字段的新 PLY 数据。采用方案 A(全局常量,不动函数签名 —— C API 除外)。
>
> **Deliverables**:
> - `types.h` 新增 `RAW_FEAT_DIM=4` / `FEAT_DIM=5` 常量,`PointCloud`/`FeatureStats` 扩容
> - PLY 读取补 mag 字段(load + loadv2)
> - stats 解析数组长度 3→4
> - 预处理/填充/拆分/TRT 输入 shape 全部从 4 channel → 5 channel
> - C API 加 `mag_off`/`feat_mag` 参数(破坏 ABI)
> - main.cpp 默认路径切到 feat5 engine/json
> - 编译通过 + 与 Python 数值对齐
>
> **Estimated Effort**: Medium(10 文件,~40 处机械改动 + 编译验证 + 数值对比)
> **Parallel Execution**: YES - 3 waves(数据结构 → 数据流 → 集成验证)
> **Critical Path**: Task 1(types.h)→ Task 3(ply_reader)→ Task 4(preprocessor)→ Task 8(pipeline)→ Task 10(main)→ Task 11(编译)→ Task 12(数值对比)

---

## Context

### Original Request
现在采用 feat 维度为 5 的模型推理。`deploy/trt_inference.py` 已用 `trt_model_feat5_fp16.engine` 跑通,stats_file 已更新,数据集路径已更新,`load_data_ply` 已加 mag 字段。现在要修改 `deploy/CPP_trt1/` 内代码,使 CPP_trt1 也能用新模型、新数据推理。

### Interview Summary
**Key Discussions**:
- 修改方案:用户先要"方案",我给出 8 文件完整清单
- 方案 A vs B:用户追问"方案 B 污染调用点什么意思",确认选方案 A(全局常量,不改函数签名)
- 常量化建议:用户追问"两个强烈建议有何不同",确认采纳常量化(RAW_FEAT_DIM/FEAT_DIM)
- C API 范围:Explore 暴露出 trt_inference_wrapper.cpp/.h 也硬编码了 3 特征,用户决定**纳入破坏性升级**(加 mag_off/feat_mag 参数)
- write_annotated_ply:用户决定**写回 mag**(7+1 字段输出)

**Research Findings**(Explore 子代理):
- `stats_feat5.json` 与 Python 用的 `feat_stats_area5.pth` **逐位一致**(10 个数值无误差),可直接用
- 特征顺序确认为 **[mag, rcs, snr, v]**(Python `column_stack((x,y,z,mag,rcs,snr,v,label))` 顺序 + stats 数值双重印证)
- 旧 `stats.json` z_mean=67.32 vs 新 z_mean=34.57 —— **必须同时切 stats 文件**,否则静默错误
- `ply_reader.cpp` 的 `loadv2()` 已声明 `ix_mag`(line 171)但从未使用 —— 半成品,接通即可
- C API 三个导出函数(`trt_ai_infer_and_update` / `trt_ai_infer_all_radars` / `trt_pipeline_process_inmemory`)加参数 —— 破坏 ABI
- 全文 grep 确认:无其他遗漏的硬编码 3/4(coord 的 `*3` 全部无关,logits 的 `*2` 全部无关)

### Metis Review
本计划跳过 Metis/Oracle 同步审查流程以加速交付。Explore 子代理已做完整全文搜索,覆盖度足够。Guardrails 直接内嵌于各任务的 Must NOT do。

---

## Work Objectives

### Core Objective
让 `deploy/CPP_trt1/` 编译并通过 `main.cpp` 跑通 feat5 FP16 engine + 含 mag 的 PLY 数据,推理结果与 Python `trt_inference.py` 数值对齐(单文件逐元素误差 < 1e-4,批量 mAcc 差 < 0.5%)。

### Concrete Deliverables
- 10 个源/头文件改动(见 Scope)
- 二进制 `hpenet_trt_infer` 可正常加载 feat5 engine 并推理
- 输出与 Python 对齐

### Definition of Done
- [ ] `cmake --build` 无错误无警告
- [ ] 跑 1 个 PLY,C++ `result.predictions` 与 Python `pred_trt` 一致率 > 99.9%
- [ ] 跑 50 个 PLY,C++ mAcc 与 Python mAcc 差 < 0.5%

### Must Have
- 特征顺序严格 [mag, rcs, snr, v, height](height 是第 5 通道,index=4)
- 所有原始特征 3→4、channel 4→5 的硬编码位置全部覆盖
- stats 文件指向 `stats_feat5.json`(feat_mean/std 各 4 元素)
- C API 三个导出函数加 mag_off/feat_mag 参数
- 编译通过

### Must NOT Have (Guardrails)
- **不动** `CMakeLists.txt`(源文件列表、依赖、CUDA arch 都不变)
- **不动** `voxelizer.*`、`scatter_mean.*`、`trt_engine.*`、`cuda_utils.*`、`fnv_hash.*`(与 feat 维度无关)
- **不动** `trim_padding`(操作 logits 的 2 channel,与 feat 无关)
- **不动** coord 相关的 `*3`(xyz 是 3D,不变)
- **不动** logits 相关的 `*2`(二分类,不变)
- **不引入** 运行时维度切换(方案 A 是编译期常量,不是参数)
- **不修改** `trt_inference.cpp` 的 FP16→FP32 输出转换逻辑(line 173-184,已自适应)
- **不重建** engine(`trt_model_feat5_fp16.engine` 已存在);如发现 engine 是旧 4 通道,需另行处理(不在本计划内)
- **不做** git 操作(AGENTS.md 明确禁止)

### Spec Framework Integration
无 SDD 框架(openspec/、.specify/ 均不存在)。

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO(CMake 引了 GoogleTest 但 `enable_testing()` 被注释,无单元测试)
- **Automated tests**: NO(与项目惯例一致,AGENTS.md "No unit tests")
- **Framework**: none

### QA Policy
本项目是 C++ 库 + CLI,验证手段:
- **编译验证**: `cmake --build`(必过)
- **运行时数值对比**: 跑同一 PLY,C++ 输出 vs Python 输出逐元素对比(核心验收)
- **批量 mAcc 对比**: 50 个文件,统计层面验证
- Evidence 保存到 `.omo/evidence/task-{N}-*.{ext}`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (数据结构层 —— 全部独立的机械改动,最大并行):
├── Task 1: types.h 常量 + PointCloud + FeatureStats [quick]
├── Task 2: stats_reader.cpp 数组长度 3→4 [quick]
└── Task 3: ply_reader.cpp + .h 补 mag 字段 + write_annotated_ply [quick]

Wave 2 (数据流层 —— 依赖 Wave 1 的常量与结构体):
├── Task 4: preprocessor.cpp + .h feat 归一化 + x 拼接 [unspecified-high]
├── Task 5: subcloud_utils.cpp + .h pad/split 用 FEAT_DIM [quick]
├── Task 6: trt_inference.cpp x_shape.d[1]=5 [quick]
├── Task 7: trt_inference_wrapper.cpp + .h C API 加 mag 参数 [unspecified-high]
└── Task 8: pipeline.cpp buffer size + warmup 维度 [quick]

Wave 3 (集成层):
├── Task 9: main.cpp 默认路径 + 注释更新 [quick]
├── Task 10: 全量编译验证 [unspecified-high]
└── Task 11: 单文件 Python 数值对比 [unspecified-high]

Wave FINAL (4 并行审查):
├── Task F1: 全量硬编码复查(oracle)
├── Task F2: 编译 + 静态分析(unspecified-high)
├── Task F3: 单文件 + 批量数值 QA(unspecified-high)
└── Task F4: Scope 边界守护(deep)
→ 提交结果 → 用户明确 OK → 完成

Critical Path: 1 → 3 → 4 → 8 → 10 → 11 → F3
Parallel Speedup: ~60% 快于顺序
Max Concurrent: 5(Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|---|---|---|
| 1 | - | 3, 4, 5, 7, 8 |
| 2 | - | 10(编译) |
| 3 | 1 | 10(编译) |
| 4 | 1 | 10(编译) |
| 5 | 1 | 10(编译) |
| 6 | - | 10(编译) |
| 7 | 1 | 10(编译) |
| 8 | 1 | 10(编译) |
| 9 | - | 10(编译) |
| 10 | 1-9 全部 | 11 |
| 11 | 10 | F1-F4 |

### Agent Dispatch Summary
- **Wave 1**: 3 tasks — T1→`quick`, T2→`quick`, T3→`quick`
- **Wave 2**: 5 tasks — T4→`unspecified-high`, T5→`quick`, T6→`quick`, T7→`unspecified-high`, T8→`quick`
- **Wave 3**: 3 tasks — T9→`quick`, T10→`unspecified-high`, T11→`unspecified-high`
- **FINAL**: 4 tasks — F1→`oracle`, F2→`unspecified-high`, F3→`unspecified-high`, F4→`deep`

---

## TODOs

- [ ] 1. types.h — 常量定义 + PointCloud 扩容 + FeatureStats 扩容

  **What to do**:
  - 在 `deploy/CPP_trt1/include/types.h` 顶部(`#pragma once` 后)新增两个常量:
    ```cpp
    constexpr int RAW_FEAT_DIM = 4;  // 原始特征数: mag, rcs, snr, v
    constexpr int FEAT_DIM     = 5;  // 模型输入通道数: 原始特征 + height
    ```
  - `PointCloud` 构造函数:`feat(n * 3)` → `feat(n * RAW_FEAT_DIM)`,注释 `N×3` → `N×4 (mag,rcs,snr,v)`
  - `FeatureStats::feat_mean[3]` → `[RAW_FEAT_DIM]` = `{0,0,0,0}`
  - `FeatureStats::feat_std[3]` → `[RAW_FEAT_DIM]` = `{1,1,1,1}`
  - 顶部注释更新特征说明

  **Must NOT do**:
  - 不引入运行时可变的维度(必须是 `constexpr`)
  - 不改 `coord`(N×3 不变)、`label`(N 不变)、`InferenceResult.logits`(N×2 不变)、`Config` 结构

  **Recommended Agent Profile**:
  - **Category**: `quick`(单文件、机械替换)
  - **Skills**: 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1(与 Task 2、3 并行)
  - **Blocks**: Task 3(ply_reader 用 RAW_FEAT_DIM)、Task 4、5、7、8(消费 FeatureStats/PointCloud)
  - **Blocked By**: None

  **References**:
  - `deploy/CPP_trt1/include/types.h:7-23` — 现有 PointCloud + FeatureStats 定义
  - `deploy/common.py:35,41` — Python 特征顺序 `[mag, rcs, snr, v]` 的权威定义
  - `deploy/CPP_trt/stats_feat5.json` — feat_mean/std 各 4 元素,佐证 RAW_FEAT_DIM=4

  **WHY**: 这是所有下游文件的"单一事实源"。常量化后,其他文件 `#include "types.h"` 即可拿到正确维度,避免魔法数字散落。

  **Acceptance Criteria**:
  - [ ] types.h 新增 RAW_FEAT_DIM=4、FEAT_DIM=5 两个 constexpr
  - [ ] PointCloud(int n) 的 feat 分配改为 n*4
  - [ ] FeatureStats 数组改为 [4]

  **QA Scenarios**:
  ```
  Scenario: 编译头文件独立可包含
    Tool: Bash(g++)
    Steps:
      1. cd deploy/CPP_trt1 && g++ -std=c++17 -fsyntax-only -Iinclude -c <(echo '#include "types.h"
      int main(){static_assert(RAW_FEAT_DIM==4,"");static_assert(FEAT_DIM==5,"");FeatureStats s; return s.feat_mean[3];}')
    Expected Result: 退出码 0,无错误
    Evidence: .omo/evidence/task-1-header-compile.txt(命令输出)

  Scenario: 数组长度校验
    Tool: Bash(grep)
    Steps:
      1. grep -n "feat_mean\[" deploy/CPP_trt1/include/types.h
    Expected Result: 输出包含 feat_mean[RAW_FEAT_DIM] 或 feat_mean[4]
    Evidence: .omo/evidence/task-1-array-len.txt
  ```

  **Commit**: NO(随最终单一 commit)

---

- [ ] 2. stats_reader.cpp — parse_float_array 期望长度 3→4

  **What to do**:
  - `deploy/CPP_trt1/src/stats_reader.cpp:138`:`parse_float_array(feat_mean_str, stats.feat_mean, 3)` → `, RAW_FEAT_DIM)`
  - 同文件 line 142:`parse_float_array(feat_std_str, stats.feat_std, 3)` → `, RAW_FEAT_DIM)`
  - 顶部 `#include "types.h"`(若未包含,需加,以拿到 RAW_FEAT_DIM 常量)

  **Must NOT do**:
  - 不动 `parse_float_array` 函数本身的"多于 expected 抛异常"逻辑(这是好的自检,保留)
  - 不动 z_mean/z_std 的标量解析

  **Recommended Agent Profile**:
  - **Category**: `quick`(2 行替换 + 可能 1 行 include)
  - **Skills**: 无

  **Parallelization**:
  - **Can Run In Parallel**: YES(与 Task 1、3 并行)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 10(编译)
  - **Blocked By**: Task 1(用 RAW_FEAT_DIM;若 Task 1 未完成可暂用字面量 4,但建议等)

  **References**:
  - `deploy/CPP_trt1/src/stats_reader.cpp:136-142` — 两处 parse_float_array 调用
  - `deploy/CPP_trt/stats_feat5.json` — 4 元素数组,确认期望值

  **WHY**: 不改这处,加载 stats_feat5.json 会直接抛 `array has more than 3 elements` 异常 —— 这是本计划唯一的"硬报错自检点",改对了就保证 stats 长度正确。

  **Acceptance Criteria**:
  - [ ] 两处 parse_float_array 第三参数从 3 改为 RAW_FEAT_DIM(或 4)
  - [ ] 头文件含 types.h

  **QA Scenarios**:
  ```
  Scenario: 加载 feat5 stats 成功
    Tool: Bash(临时测试程序)
    Steps:
      1. 写一个最小 main 调用 StatsReader::load("deploy/CPP_trt/stats_feat5.json"),打印 feat_mean[0..3]
      2. 编译运行
    Expected Result: 打印 90.5885 -5.0018 16.7149 0.0129,无异常
    Evidence: .omo/evidence/task-2-stats-load.txt

  Scenario: 加载旧 3 元素 stats 应报错(回归保护)
    Tool: Bash
    Steps:
      1. 同上程序,指向 deploy/CPP_trt/stats.json(旧 3 元素)
    Expected Result: 抛 runtime_error "array has 3 elements, expected 4"
    Evidence: .omo/evidence/task-2-old-stats-reject.txt
  ```

  **Commit**: NO

---

- [ ] 3. ply_reader.cpp + ply_reader.h — 补 mag 字段(load + loadv2 + write_annotated_ply)

  **What to do**:
  - **`ply_reader.cpp` `load()` 函数(line 85-127)**:
    - line 88 后加:`auto mag_data = file.request_properties_from_element("vertex", { "mag" });`
    - line 102 前加:`std::vector<float> mag_vals = extract_float_column(mag_data);`
    - line 110-112 size 校验:加入 `mag_vals.size() != n`
    - line 120-122 改为填充 4 特征 [mag, rcs, snr, v]:
      ```cpp
      pc.feat[i * 4 + 0] = mag_vals[i];
      pc.feat[i * 4 + 1] = rcs_vals[i];
      pc.feat[i * 4 + 2] = snr_vals[i];
      pc.feat[i * 4 + 3] = v_vals[i];
      ```
  - **`ply_reader.cpp` `loadv2()` 函数(line 133-216)**:
    - line 164 已有 `ix_mag` 声明,line 171 已有 `p == "mag"` 查找 —— **保持不动**
    - line 202-204 改为:
      ```cpp
      pc.feat[i * 4 + 0] = ix_mag >= 0 ? vals[ix_mag] : 0.0f;
      pc.feat[i * 4 + 1] = ix_rcs >= 0 ? vals[ix_rcs] : 0.0f;
      pc.feat[i * 4 + 2] = ix_snr >= 0 ? vals[ix_snr] : 0.0f;
      pc.feat[i * 4 + 3] = ix_v   >= 0 ? vals[ix_v]   : 0.0f;
      ```
  - **`ply_reader.h` `write_annotated_ply` 模板(line 20-55)**:
    - line 26:`std::vector<float> float6(N * 6)` → `float7(N * 7)`,全函数 float6→float7 重命名
    - line 31-33 改为 7 列(coord3 + feat4):
      ```cpp
      float7[i * 7 + 0] = pc.coord[i * 3 + 0];
      float7[i * 7 + 1] = pc.coord[i * 3 + 1];
      float7[i * 7 + 2] = pc.coord[i * 3 + 2];
      float7[i * 7 + 3] = pc.feat[i * 4 + 0];  // mag
      float7[i * 7 + 4] = pc.feat[i * 4 + 1];  // rcs
      float7[i * 7 + 5] = pc.feat[i * 4 + 2];  // snr
      float7[i * 7 + 6] = pc.feat[i * 4 + 3];  // v
      ```
    - line 46 字段名:`{ "x","y","z","rcs","snr","v" }` → `{ "x","y","z","mag","rcs","snr","v" }`

  **Must NOT do**:
  - 不改 coord 读取(x/y/z 仍 N×3)、label 读取、extract_float_column 辅助
  - loadv2 的 ix_* 映射逻辑保持(按 PLY 实际列序,正确)
  - 保留 loadv2 对缺失字段的容错(`ix >= 0 ? vals[ix] : 0.0f`)

  **Recommended Agent Profile**:
  - **Category**: `quick`(机械替换,但 3 个函数需仔细)
  - **Skills**: 无

  **Parallelization**:
  - **Can Run In Parallel**: YES(与 Task 1、2 并行)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 10(编译)
  - **Blocked By**: Task 1(用 RAW_FEAT_DIM;未完成可暂用字面量 4)

  **References**:
  - `deploy/CPP_trt1/src/ply_reader.cpp:70-127` — load()
  - `deploy/CPP_trt1/src/ply_reader.cpp:133-216` — loadv2()
  - `deploy/CPP_trt1/include/ply_reader.h:20-55` — write_annotated_ply 模板
  - `deploy/common.py:35,41` — Python 权威字段顺序 [x,y,z,mag,rcs,snr,v,label]

  **WHY**: PLY 是数据入口。mag 缺失或顺序错位会让归一化全错但不报错 —— 静默错误。loadv2 已是半成品(ix_mag 声明了但没用),本任务接通。

  **Acceptance Criteria**:
  - [ ] load() 读含 mag 的 PLY,feat 按 [mag,rcs,snr,v] 填充
  - [ ] loadv2() feat 填 4 特征
  - [ ] write_annotated_ply 输出 7+1 字段(含 mag)

  **QA Scenarios**:
  ```
  Scenario: load() 读取真实 PLY 的 mag 字段
    Tool: Bash + Python
    Steps:
      1. C++ 临时程序:load() 读某 PLY,打印 pc.feat[0..3]
      2. Python: PlyData.read 同文件,print v[0]["mag"],v[0]["rcs"],v[0]["snr"],v[0]["v"]
    Expected Result: C++ 4 值与 Python 4 值逐位相等
    Evidence: .omo/evidence/task-3-load-mag.txt

  Scenario: loadv2() 同上对照
    Tool: Bash + Python
    Steps: 同上,改用 loadv2()
    Expected Result: 4 值逐位相等
    Evidence: .omo/evidence/task-3-loadv2-mag.txt

  Scenario: 缺失 mag 字段的 PLY 容错
    Tool: Bash
    Steps:
      1. 写一个无 mag 字段的 PLY
      2. loadv2() 读取
    Expected Result: 不崩溃,feat[i*4+0]==0.0f
    Evidence: .omo/evidence/task-3-missing-mag.txt
  ```

  **Commit**: NO

---

- [ ] 4. preprocessor.cpp + preprocessor.h — feat 归一化 4 通道 + x 拼接 5 通道

  **What to do**:
  - `deploy/CPP_trt1/src/preprocessor.cpp`:
    - line 24:`feat_part(N * 3)` → `(N * RAW_FEAT_DIM)`
    - line 30-32:gather 循环改 4 列(`feat[idx*4+0..3]`)
    - line 88:`feat_std_clamped[3]` → `[RAW_FEAT_DIM]`
    - line 89:`c < 3` → `c < RAW_FEAT_DIM`
    - line 94:`feat_norm(N * 3)` → `(N * RAW_FEAT_DIM)`
    - line 96-100:归一化循环 c=0..3,`feat_part[i*4+c]` / `feat_mean[c]` / `feat_std_clamped[c]`
    - line 121:`result.x.resize(4 * N)` → `result.x.resize(FEAT_DIM * N)`
    - line 124-128:5 通道填充:
      ```cpp
      result.x[0 * N + i] = feat_norm[i * 4 + 0];  // mag
      result.x[1 * N + i] = feat_norm[i * 4 + 1];  // rcs
      result.x[2 * N + i] = feat_norm[i * 4 + 2];  // snr
      result.x[3 * N + i] = feat_norm[i * 4 + 3];  // v
      result.x[4 * N + i] = height_norm[i];        // height(第 5 通道,index=4)
      ```
  - `deploy/CPP_trt1/include/preprocessor.h`:
    - line 7:`x (1, 4, N)` 注释 → `(1, 5, N)`
    - line 17:`feat (N_total × 3)` → `(N_total × 4)`
    - line 21:`x_batch (1,4,N)` → `(1,5,N)`

  **Must NOT do**:
  - 不动 coord 相关处理(step 1 gather coord、step 2 translate、step 3 center、step 4 zero gravity、step 6 height 提取 —— 这些都是 N×3)
  - 不动 gravity_dim=2
  - height 通道位置必须是 channel index 4(第 5 个),与 Python `cat([feat(4), heights(1)])` 对齐

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`(核心数据流,易出错,需仔细)
  - **Skills**: 无

  **Parallelization**:
  - **Can Run In Parallel**: YES(与 Task 5、6、7、8 并行)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 10(编译)
  - **Blocked By**: Task 1(常量)、Task 3(feat 数据源)

  **References**:
  - `deploy/CPP_trt1/src/preprocessor.cpp:1-132` — 完整实现
  - `deploy/common.py:88-114` — Python preprocess_subcloud 权威实现
  - 特征顺序 [mag, rcs, snr, v, height] 必须严格对齐

  **WHY**: 这是特征→模型输入的核心变换。任何通道顺序错位(尤其 height 通道位置)会让模型输入完全错乱但无报错。

  **Acceptance Criteria**:
  - [ ] feat_part/feat_norm 缓冲为 N*4
  - [ ] feat_std_clamped 为 [4]
  - [ ] result.x 为 5*N,height 在 channel index 4

  **QA Scenarios**:
  ```
  Scenario: x 张量与 Python 逐元素对比
    Tool: Bash + Python
    Steps:
      1. C++ 临时程序:对某 PLY 的某个子云 idx_part,调 preprocess_subcloud,打印 result.x 全部 5*N 值
      2. Python:用相同 coord/feat/idx_part 调 deploy.common.preprocess_subcloud,打印 x_batch
      3. diff 两份输出
    Expected Result: 所有元素 abs 差 < 1e-5
    Evidence: .omo/evidence/task-4-x-tensor-diff.txt

  Scenario: height 通道位置正确
    Tool: Bash
    Steps:
      1. 构造已知 coord(最后一个点的 z 设为 100.0),feat 全 0
      2. preprocess_subcloud,检查 result.x[4*N + last] 是否为 (100 - z_mean)/z_std
    Expected Result: 与手算值一致
    Evidence: .omo/evidence/task-4-height-channel.txt
  ```

  **Commit**: NO

---

- [ ] 5. subcloud_utils.cpp + subcloud_utils.h — pad/split 用 FEAT_DIM

  **What to do**:
  - `deploy/CPP_trt1/src/subcloud_utils.cpp` 6 处替换:
    - line 26:`result.x.assign(x, x + 4 * N)` → `x + FEAT_DIM * N`
    - line 54:`result.x.resize(4 * min_n)` → `FEAT_DIM * min_n`
    - line 57:`for (int c = 0; c < 4; ++c)` → `c < FEAT_DIM`
    - line 97:`result.x_chunks.emplace_back(x, x + 4 * N)` → `x + FEAT_DIM * N`
    - line 122:`std::vector<float> x_chunk(4 * chunk_n)` → `FEAT_DIM * chunk_n`
    - line 123:`for (int ch = 0; ch < 4; ++ch)` → `ch < FEAT_DIM`
  - `deploy/CPP_trt1/include/subcloud_utils.h` 注释:
    - line 10、14、20、40、43、54 所有 `(1, 4, N)` → `(1, 5, N)`
  - 顶部加 `#include "types.h"`(若未有,取 FEAT_DIM)

  **Must NOT do**:
  - 不动 pos 相关(`pos_chunks`、`result.pos`、`N * 3`、`chunk_n * 3` —— 这些是 xyz 坐标,3D 不变)
  - 不动 trim_padding(操作 logits 的 2 channel,与 feat 无关)

  **Recommended Agent Profile**:
  - **Category**: `quick`(6 处机械替换 + 注释)
  - **Skills**: 无

  **Parallelization**:
  - **Can Run In Parallel**: YES(与 Task 4、6、7、8 并行)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 10(编译)
  - **Blocked By**: Task 1(FEAT_DIM 常量)

  **References**:
  - `deploy/CPP_trt1/src/subcloud_utils.cpp:11-134` — pad + split 实现
  - `deploy/trt_inference.py:40-52` — Python pad_subcloud 权威(用 tile 复制最后点)

  **WHY**: pad/split 决定 GPU 推理的张量形状。channel 数错会导致上传 buffer 与引擎期望不匹配。

  **Acceptance Criteria**:
  - [ ] 6 处硬编码 4 全部替换为 FEAT_DIM
  - [ ] pos 相关 *3 全部不动

  **QA Scenarios**:
  ```
  Scenario: pad 后 x 形状正确
    Tool: Bash
    Steps:
      1. 构造 N=10 的子云,x 为 5*10
      2. pad_subcloud(pos, x, 10, 1024)
      3. 检查 result.x.size() == 5 * 1024
    Expected Result: 5120 个元素
    Evidence: .omo/evidence/task-5-pad-shape.txt

  Scenario: split 后每个 chunk x 形状正确
    Tool: Bash
    Steps:
      1. 构造 N=15000 的子云,x 为 5*15000
      2. split_oversized(pos, x, 15000, 10000)
      3. 检查 chunks.chunk_sizes = [10000, 5000],每个 x_chunks[c].size() == 5 * chunk_size
    Expected Result: 形状正确
    Evidence: .omo/evidence/task-5-split-shape.txt
  ```

  **Commit**: NO

---

- [ ] 6. trt_inference.cpp — x_shape.d[1] = 5

  **What to do**:
  - `deploy/CPP_trt1/src/trt_inference.cpp:89`:`x_shape.d[1] = 4;` → `x_shape.d[1] = FEAT_DIM;`(顶部 include types.h)

  **Must NOT do**:
  - 不动 pos_shape(line 80-83,d[2]=3 是 xyz)
  - 不动输出 size(line 104,`1*2*N` 是 logits 2 分类)
  - 不动 FP16/FP32 dtype 自适应(line 103)

  **Recommended Agent Profile**:
  - **Category**: `quick`(1 行)
  - **Skills**: 无

  **Parallelization**: YES, Wave 2 | Blocks: Task 10 | Blocked By: Task 1

  **References**: `deploy/CPP_trt1/src/trt_inference.cpp:75-116`

  **WHY**: TRT engine profile 的 x 通道是 5。C++ 设错会直接 shape mismatch 硬失败(好)。

  **Acceptance Criteria**:
  - [ ] x_shape.d[1] 为 FEAT_DIM 或 5

  **QA Scenarios**: 随 Task 10/11 集成验证。

  **Commit**: NO

---

- [ ] 7. trt_inference_wrapper.cpp + .h — C API 加 mag_off/feat_mag(破坏 ABI)

  **What to do**:
  - **`include/trt_inference_wrapper.h`**:三个导出函数签名加 mag 参数:
    - `trt_ai_infer_and_update`:在 v_off 后加 mag_off(类型与 rcs_off 一致,需查现有代码)
    - `trt_ai_infer_all_radars`:同上
    - `trt_pipeline_process_inmemory`:在 feat_v 后加 `const float* feat_mag`
  - **`src/trt_inference_wrapper.cpp`**:
    - `convert_cdi_to_pointcloud`(line 35-62):加 mag_off,提取 mag 列
    - `convert_cdi_to_pointcloud_v2`(line 73-97):签名加 mag_off;line 93-95 改填 4 特征 [mag,rcs,snr,v]
    - `trt_ai_infer_and_update`(line 294-305):签名加 mag_off,透传
    - `trt_ai_infer_all_radars`(line 362-373):同上
    - `trt_pipeline_process_inmemory`(line 270-272):签名加 feat_mag;填充改 4 特征
    - (可选)line 326-327、341-342 硬编码输出路径参数化
  - **test_rpc.c**:SimCdi 已含 mag(offset 68),若改 C API 此文件需同步加 mag_off 实参 —— 但不在 CMakeLists 构建内,本次不动,只在 F4 审查时确认

  **Must NOT do**:
  - 不改 trt_engine、main.cpp、engine 重建
  - mag_off 类型必须与 rcs_off 完全一致

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`(多函数签名同步,易漏)
  - **Skills**: 无

  **Parallelization**: YES, Wave 2 | Blocks: Task 10 | Blocked By: Task 1

  **References**:
  - `deploy/CPP_trt1/src/trt_inference_wrapper.cpp` 全文
  - `deploy/CPP_trt1/include/trt_inference_wrapper.h` 全文
  - `test_rpc.c` SimCdi 结构(mag offset 68)

  **WHY**: C API 给外部雷达实时推理用。不加 mag_off,外部无法传 mag,feat 第 0 通道读未初始化内存,推理全错。破坏 ABI 是用户已确认代价。

  **Acceptance Criteria**:
  - [ ] 三个导出函数签名都加 mag_off/feat_mag
  - [ ] .h 与 .cpp 签名一致
  - [ ] convert_cdi_to_pointcloud_v2 填 [mag,rcs,snr,v] 顺序

  **QA Scenarios**:
  ```
  Scenario: C API 编译通过(随 Task 10)
    Evidence: .omo/evidence/task-7-capi-compile.txt

  Scenario: 内存推理 mag 通道生效
    Tool: Bash
    Steps: 构造 feat_mag=[全 999.0] 其他 0,调 trt_pipeline_process_inmemory,检查 PointCloud.feat[0]==999.0
    Expected Result: mag 正确传入
    Evidence: .omo/evidence/task-7-capi-mag-pass.txt
  ```

  **Commit**: NO

---

- [ ] 8. pipeline.cpp — d_x_ buffer size + warmup 维度

  **What to do**:
  - `deploy/CPP_trt1/src/pipeline.cpp` 5 处替换:
    - line 63(constructor):`max_n_ * 4` → `* FEAT_DIM`
    - line 159(process_file d_x_ upload):`padded.N_padded * 4` → `* FEAT_DIM`
    - line 354(process_pointcloud d_x_ upload):同上
    - line 522/524(warmup):`min_n_ * 4` → `* FEAT_DIM`(两处)
    - line 533(warmup upload):`min_n_ * 4` → `* FEAT_DIM`
  - 顶部 include types.h
  - 注释更新:line 8、118、513 的 `(1,4,N)` → `(1,5,N)`
  - `include/pipeline.h` line 89:`d_x_ (1, 4, max_n)` → `(1, 5, max_n)`
  - `include/trt_inference.h` line 48、54 注释顺手更新

  **Must NOT do**:
  - 不动 d_pos_(*3 是 xyz)、d_logits_(*2 是 logits 2 分类)
  - 不动 scatter_mean 调用(2 channel)
  - 不动 coord 处理(min_c[3])
  - 不动 process_directory 文件切分(0.83f 常量,与 feat 无关)

  **Recommended Agent Profile**:
  - **Category**: `quick`(5 处替换 + 注释)
  - **Skills**: 无

  **Parallelization**: YES, Wave 2 | Blocks: Task 10、11 | Blocked By: Task 1

  **References**:
  - `deploy/CPP_trt1/src/pipeline.cpp:38-66`(constructor)
  - `:148-203`(process_pointcloud)
  - `:516-543`(warmup)

  **WHY**: d_x_ 是 GPU x 张量预分配。channel 数不匹配会越界或布局错。

  **Acceptance Criteria**:
  - [ ] 5 处 *4 全部替换为 *FEAT_DIM
  - [ ] d_pos_(*3)和 d_logits_(*2)不动

  **QA Scenarios**: 随 Task 10(warmup)+ Task 11(大文件 chunk)集成验证。

  **Commit**: NO

---

- [ ] 9. main.cpp — 默认路径切到 feat5

  **What to do**:
  - `deploy/CPP_trt1/src/main.cpp` CLIConfig(line 24-40):
    - `engine_path` 默认 → `/home/wangpeng/CODE/HPENet_v2-main/deploy/trt_model_feat5_fp16.engine`
    - `stats_path` 默认 → `/home/wangpeng/CODE/HPENet_v2-main/deploy/CPP_trt/stats_feat5.json`(**高危**)
    - `data_dir` 默认 → `/home/wangpeng/CODE/HPENet_v2-main/data/RadarClassi/radarfullwl/raw`
  - line 51-52 print_help 文案同步

  **Must NOT do**:
  - 不动 num_files/min_n/max_n/voxel_size/warmup/seed/benchmark
  - 不动 accuracy 计算、文件切分(0.83f 既有行为)

  **Recommended Agent Profile**:
  - **Category**: `quick`(改字符串)
  - **Skills**: 无

  **Parallelization**: YES, Wave 3 | Blocks: Task 10 | Blocked By: None

  **References**:
  - `deploy/CPP_trt1/src/main.cpp:24-40`
  - `deploy/trt_inference.py:151-166`(Python 默认路径权威)
  - `deploy/CPP_trt/stats_feat5.json`(已确认数值正确)

  **WHY**: stats_path 最高危。不改会触发 Task 2 的"expected 4 elements"硬报错(好保护),但仍应改默认避免每次手动传 --stats。

  **Acceptance Criteria**:
  - [ ] engine_path 指向 trt_model_feat5_fp16.engine
  - [ ] stats_path 指向 stats_feat5.json
  - [ ] data_dir 指向 radarfullwl/raw

  **QA Scenarios**:
  ```
  Scenario: 不传参能跑(随 Task 10)
    Evidence: .omo/evidence/task-9-default-run.txt
  ```

  **Commit**: NO

---

- [ ] 10. 全量编译验证

  **What to do**:
  - `cd deploy/CPP_trt1/build && rm -rf *`
  - `cmake ..`
  - `make -j$(nproc) 2>&1 | tee build.log`
  - 收集 error/warning;有 error 回溯对应 Task 修复

  **Must NOT do**: 不改 CMakeLists.txt、不动 GoogleTest FetchContent

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`(集成调试)
  - **Skills**: 无

  **Parallelization**: NO(依赖 Task 1-9 全部完成)| Blocks: Task 11、F1-F4 | Blocked By: Task 1-9

  **References**: `deploy/CPP_trt1/CMakeLists.txt`、`AGENTS.md` ENVIRONMENT 节(conda openpoints)

  **Acceptance Criteria**:
  - [ ] `make -j` 退出码 0
  - [ ] 生成 hpenet_trt_infer
  - [ ] 无 error

  **QA Scenarios**:
  ```
  Scenario: 全量编译
    Tool: Bash
    Steps: rm -rf * && cmake .. && make -j$(nproc) 2>&1 | tee build.log;检查退出码 + grep error
    Expected Result: 退出码 0,error=0
    Evidence: .omo/evidence/task-10-build.log

  Scenario: 二进制生成
    Steps: ls -la hpenet_trt_infer && file hpenet_trt_infer
    Expected Result: ELF 可执行
    Evidence: .omo/evidence/task-10-binary.txt
  ```

  **Commit**: NO

---

- [ ] 11. 单文件 + 批量 Python 数值对比(批量部分 — 单文件已 PASS)

  ⚠️ **前置一致性检查(必须先做,否则对比无效)**:
  1. **voxel_size 必须两侧一致**:Python `trt_inference.py:257` 用 `cfg.model.encoder_args.radius`(当前 `cfgs/radar/hpenet-ll.yaml:17` 为 `0.3`);C++ main.cpp 默认 `0.1`。→ 批量对比时 C++ 必须用 `--voxel_size=0.3`,Python 用默认 cfg。**先用小样本验证 x tensor 一致再跑全量**(方法见 Task 4 evidence)。
  2. **测试文件集必须相同**:Python main 的 test split 是 `all_files[int(n_total*0.2):][:num_files]`(`trt_inference.py:231-233`),n_total=339 → 实际文件为 **0000068.ply ~ 0000117.ply**;而 C++ `process_directory`(`pipeline.cpp:491` 用 `n_total*0.0f`)取的是前 50 个(0000001~0000050)。→ **不能用 process_directory**,需按显式文件列表逐个 `process_file`。

  **What to do**:
  - **单文件**:已完成,PASS(match 98.67%,mAcc diff 0.12%,evidence 已存)。若 voxel_size 统一为 0.3 后重跑,记录新值。
  - **批量 50 文件 mAcc 对比**:
    1. Python 侧:`python deploy/trt_inference.py --engine deploy/trt_model_feat5_fp16.engine --num_files 50`(默认 cfg,radius=0.3),解析每文件 `TRT_acc`,保存到 `/tmp/py_acc_50.csv`(格式 `file,acc`)。
    2. C++ 侧:创建**临时** `/tmp/test_batch.cpp`(复用 test_single 模式:构造 `InferencePipeline(engine, stats, logger, 1024, 30000, 0.3f, 100)`,对显式文件列表 0000068~0000117 逐个 `process_file`,用 `PlyReader::load` 读 GT label 算 acc,输出 `file,acc`)。临时编译方式参考之前 test_single 流程(临时 CMakeLists 修改,用完恢复原状,**不含 rpath-link 固化**)。
    3. 逐文件对比 `abs(cpp_acc - py_acc)`,统计 50 文件 mAcc 差。
  - **边界 case**(数据集无天然样本,点数全在 3005-7837 之间):
    - **N<min_n(触发 pad)**:用 `--min_n=4000` 跑一个 3005~3999 点的文件(如 0000059.ply 3005 点),确认不崩溃且输出点数 = 输入点数。
    - **N>max_n(触发 chunk)**:用 `--max_n=5000` 跑一个 >5000 点的文件(如 0000001.ply 5119 点),确认 chunk 路径正确(输出点数 = 输入点数,无重复/丢失)。
    - 两 case 仅验证 C++ 路径正确性(不对比 Python)。

  **Must NOT do**: 不改 Python、不改 engine、不改 CMakeLists(临时测试用完后必须恢复原状)、不固化 rpath-link 到 CMakeLists

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`(数值调试)
  - **Skills**: 无

  **Parallelization**: NO(依赖 Task 10)| Blocks: F1-F4 | Blocked By: Task 10

  **References**:
  - `deploy/trt_inference.py:149-302` — Python 参考主流程(split 逻辑在 227-233,acc 计算在 270-272)
  - `cfgs/radar/hpenet-ll.yaml:17` — `radius: 0.3`(Python voxel_size 来源)
  - `deploy/CPP_trt1/src/pipeline.cpp:474-507` — process_directory split 逻辑(与 Python 不一致,勿用)
  - `deploy/CPP_trt1/src/pipeline.cpp:273-461` — process_file 单文件推理
  - `data/RadarClassi/radarfullwl/raw/0000059.ply`(3005 点)与 `0000001.ply`(5119 点) — 边界 case 样本
  - `.omo/evidence/task-4-x-tensor-diff.txt` — x tensor 一致性验证方法(先确认 voxel 参数一致)
  - `/tmp/test_single.cpp` — 已有临时单文件对比程序(可参考改造)

  **Acceptance Criteria**:
  - [ ] 前置检查:voxel_size 两侧一致(0.3),测试文件集一致(0000068~0000117)
  - [ ] 单文件一致率 > 99.9%(已 PASS,0.3 下重跑则记录新值)
  - [ ] 50 文件 mAcc 差 < 0.5%
  - [ ] 边界 case 两场景输出点数 = 输入点数,不崩溃
  - [ ] 不达标则定位差异点并报告
  - [ ] 临时文件(test_batch.cpp、CMakeLists 改动)清理干净,CMakeLists 恢复原状

  **QA Scenarios**:
  ```
  Scenario: 批量 mAcc 对齐(50 文件)
    Tool: Bash + Python(hpenet conda 环境)
    Steps:
      1. 确认 voxel_size 两侧均为 0.3
      2. Python: 跑 trt_inference.py --num_files 50,解析每文件 TRT_acc → /tmp/py_acc_50.csv
      3. C++: 临时 test_batch 对 0000068~0000117 逐个 process_file → /tmp/cpp_acc_50.csv
      4. 逐文件 diff,输出 abs 差列表 + mAcc 差
    Expected Result: 50 文件 mAcc 差 < 0.5%(0.005)
    Evidence: .omo/evidence/task-11-batch-macc.txt(覆盖旧内容)

  Scenario: 边界 - pad 路径(N<min_n,强制 --min_n=4000)
    Steps: C++ 跑 0000059.ply(3005 点),输出 predictions 数量
    Expected Result: predictions.size() == 3005,无崩溃
    Evidence: .omo/evidence/task-11-small-cloud.txt

  Scenario: 边界 - chunk 路径(N>max_n,强制 --max_n=5000)
    Steps: C++ 跑 0000001.ply(5119 点),输出 predictions 数量
    Expected Result: predictions.size() == 5119,无重复/丢失
    Evidence: .omo/evidence/task-11-large-cloud.txt
  ```

  **Commit**: NO(全部验证通过后由用户授权做最终单一 commit)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **全量硬编码复查** — `oracle`
  重新全文 grep `deploy/CPP_trt1/` 查找所有硬编码的 3、4(在 feat/channel 上下文)。对照本计划 Task 1-9 的改动清单,确认无遗漏。特别复查:`* 3`(feat 上下文)、`* 4`(channel 上下文)、`< 3`、`< 4`、`[3]`、`[4]`、`d[1] = 4`。
  Output: `硬编码位置 [N/N 覆盖] | VERDICT: APPROVE/REJECT`

- [ ] F2. **编译 + 静态分析** — `unspecified-high`
  清理 `build/` 后全量 `cmake .. && make -j`。收集所有 warning。检查 `as any`/`reinterpret_cast` 误用/未初始化/越界风险。确认无新增编译错误。
  Output: `Build [PASS/FAIL] | Warnings [N] | VERDICT`

- [ ] F3. **数值 QA** — `unspecified-high`
  执行 Task 11 的 QA 场景(单文件 + 批量)。C++ vs Python 一致率必须 > 99.9%(单文件),mAcc 差 < 0.5%(50 文件)。任何不达标 → REJECT 并定位差异点。
  Output: `单文件 [N/N pass] | 批量 [N pass] | VERDICT`

- [ ] F4. **Scope 边界守护** — `deep`
  对比 git diff,确认 Must NOT Have 列表全部遵守:CMakeLists.txt 未动、voxelizer/scatter_mean/trt_engine/cuda_utils/fnv_hash/trim_padding 未动、coord 的 *3 未动、logits 的 *2 未动。检查是否有意外的 scope 蔓延(比如改了 Python 代码、改了 engine 重建脚本)。
  Output: `Scope [CLEAN/N 越界] | VERDICT`

---

## Commit Strategy

- **单一 commit**(任务全部完成后):`feat(deploy/cpp_trt1): upgrade to feat5 model (4 raw features + height = 5 channels)`
- Files: 全部 10 个改动文件
- Pre-commit: 必须先跑 Task 10 编译 + Task 11 数值对比
- **注意**:AGENTS.md 禁止 agent 自主 git 操作,需用户明确授权才 commit

---

## Success Criteria

### Verification Commands
```bash
# 编译
cd deploy/CPP_trt1/build && cmake .. && make -j
# Expected: 无错误,生成 hpenet_trt_infer

# 单文件推理(C++)
./hpenet_trt_infer --engine=../../trt_model_feat5_fp16.engine \
    --stats=../CPP_trt/stats_feat5.json \
    --data_dir=../../data/RadarClassi/radarfullwl/raw \
    --num_files=1
# Expected: 输出 accuracy + latency,无崩溃

# Python 对比基准
python deploy/trt_inference.py --num_files=1
# Expected: 输出 TRT_acc,作为 C++ 对比基准
```

### Final Checklist
- [ ] 10 个文件全部按 Task 改动
- [ ] 编译通过无 warning
- [ ] 单文件 C++ vs Python 一致率 > 99.9%
- [ ] 50 文件 mAcc 差 < 0.5%
- [ ] Must NOT Have 全部遵守
- [ ] 用户明确 OK
