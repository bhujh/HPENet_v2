# Task-7 证据：plugin.md 文档更新（§14 v15.0 三算法 FPS 插件族）

- 时间: 2026-08-19 06:09
- 环境: 文档 = `plugin.md`（仓库根，74KB）| 实测数据源 = `.omo/evidence/task-{1..6}{,b}-fps-samplefps-flashfps.{log,md}` | 验证机 = L20（数据复核不重跑，交叉引用既有 evidence）
- 交付物: `plugin.md` §14「v15 增补任务：三算法 FPS 插件族（FPS / SampleFPS / FlashFPS）」v15.0 完整章节

## 1. 交付物位置与章节结构

- **文件**: `plugin.md`，§14 位于 L1440-1699（共 260 行），版本标注 v15.0（§14.11 实施记录，2026-08-19）
- **章节结构**（11 小节）：

| 节 | 标题 | 内容要点 |
|---|---|---|
| §14 总纲 | 目标 + 结论先行 | fps_algo 三开关、计划代号、evidence 文件命名约定（`.omo/evidence/task-{N}-fps-samplefps-flashfps.{log,md}`） |
| 14.1 | 三算法设计总览 | 4 插件 type/version 表、接口约定表（属性/输出形状/format/workspace/enqueue/序列化/Creator）、新增文件清单、未动现役文件声明 |
| 14.2 | SampleFPS 自研精确 kernel | 两阶段结构（build+iterate）、CSR scratch 布局、四步迭代、float32 左结合距离表达式、QA 修复的 3 个 bug、验收判据 |
| 14.3 | FlashFPS Prune 语义 + 图级 Cache | keep_rate 语义、num_points 公式、升序填充、图级 Cache（stage1→FlashFPS / stage2-4→PrefixFPS） |
| 14.4 | PrefixFPS Cache 载体 | workspace=0、动态形状 kFLOOR_DIV、torch forward (B,M)、第 8 个插件 |
| 14.5 | fps_algo 开关与导出接线 | 三档替换表、判据（nn.Sequential 子模块命名）、回归保证、hpenet 域节点清单 |
| 14.6 | 前缀等价性论证 | 数学论证 + cache-only 直接实证（N=4096/5500 双档逐索引）+ keep_rate≥0.5 约束 |
| 14.7 | FlashFPS 语义移植说明 | 与上游仓库/论文差异表（升序填充 vs 随机填充标注） |
| 14.8 | 对比实测数据回填 | 单算子对拍（14.8.1）、acc（14.8.2）、延迟（14.8.3）、logits 对拍（14.8.4） |
| 14.9 | 门槛判定与瓶颈分析 | 8 条门槛判定表 + FPS<5ms 未达的单 block 结构瓶颈分析 |
| 14.10 | 现役 v14 profile 隐患 | max_n=6500 < 子云最大 6988 的独立发现 + 重导建议 |
| 14.11 | 实施记录（v15.0） | 落地文件、6 个实施中问题、验证结果汇总、遗留项 |

## 2. 实测数据来源（task-N evidence 交叉引用）

§14 全部实测数据均来自既有 evidence 文件，未新造数字；引用映射如下：

| plugin.md 章节 | 数据来源 evidence | 核对要点 |
|---|---|---|
| §14.2 验收判据 / §14.11 问题 1-3 | `task-1-fps-samplefps-flashfps.log` | 3 个 bug（bbox 原子写反 / float 原子无重载 / 多 batch 切片）+ tie-free 100% 逐索引一致 |
| §14.2 接口表 / §14.4 | `task-2-fps-samplefps-flashfps.log` | SampleFPS 插件属性、符号 nm 验证、N=1024/5500 对拍矩阵 |
| §14.3-14.5 接线 / §14.4 动态形状 | `task-3-fps-samplefps-flashfps.md` | PrefixFPS 插件注册、fps 回归 one-liner、动态 N=1024/6500 断言 |
| §14.3 prune / §14.7 语义差异 | `task-4-fps-samplefps-flashfps.log` | FlashFPS 三件套、prune 语义独立验证（k∈{0.75,0.5,0.25} 全 prefix_exact/tail_ascending） |
| §14.6 cache-only 实证 / §14.5 接线 | `task-4b-fps-samplefps-flashfps.md` | 1×FlashFPS+3×PrefixFPS 图、CACHE-ONLY SEMANTICS N=4096/5500 逐索引、keep_rate·M₁≥M₂ |
| §14.8.1 单算子对拍 | `task-5-fps-samplefps-flashfps.md`（及 task-2/4 log 的同一中位数口径） | idx/seq 100%、mismatch=0、tie=none；vs 现役 100% |
| §14.8.2 acc / §14.8.3 延迟 / §14.8.4 / §14.9 | `task-6-fps-samplefps-flashfps.md` | ti10=0.9741、modetest 58 文件、端到端延迟、nsys FPS 段、平局归因、档位阶梯 |
| §14.10 profile 隐患 | `task-6-fps-samplefps-flashfps.md` §0 | max_sub=6988 > max_n=6500，现役 v14 engine 越 profile 产出垃圾 |

## 3. 验证方式（grep 无悬空引用 + 数据逐项核对）

### 3.1 引用完整性（grep）
- plugin.md §14（L1440-1700）中所有 task-N evidence 散文引用（task-1/2/3/4/4b/5/6）逐一映射到 `.omo/evidence/` 实际文件：**7 个引用目标全部存在**（task-1/2/4 为 .log，task-3/4b/5/6 为 .md），无悬空引用
- 文件命名约定句「`.omo/evidence/task-{N}-fps-samplefps-flashfps.{log,md}`」与实际 evidence 文件命名一致
- 交叉引用断言与来源文件一致：task-3 的「动态 N=1024/6500 (1,256)/(1,1625)」、task-4b 的「CACHE-ONLY N=4096/5500 M=687/343」均逐项核对存在

### 3.2 数据逐项核对（抽查全部关键数字，非重跑）
- **acc**（§14.8.2 vs task-6 §4.1/4.2）：ti10 现役/samplefps/cache-only=0.9741、fps=0.9741、k0.75=0.9707、k0.5=0.9612、k0.25=0.9535；modetest fps=0.9321、samplefps=0.9321、k0.75=0.9338（反超 +0.0017）——**全部一致**
- **延迟**（§14.8.3 vs task-6 §6）：端到端 median fps 28.8/samplefps 134.5/cache-only 61.7/k0.75 43.7ms；nsys FPS 段 fps 105.7ms-61.0%-8.1ms / samplefps 1108.4ms-97.0%-85.3ms / k0.75 299.4ms-82.5%-23.0ms、-72%——**全部一致**
- **单算子**（§14.8.1 vs task-5/2/4）：idx/seq 100%、mismatch=0、tie=none；k0.75 N=5500 24.677ms（task-4 log 同值）——**正确性断言一致**；median 计时值 run-to-run 有微小波动（task-5 重跑 N=1024 由 5.715→1.379ms，属同一语义下测量噪声，非数据错误）
- **平局归因**（§14.8.4 vs task-6 §3）：idx_mismatch_rate=0.0021%、mean idx_match=99.9981%、seq_exact=100.0000%、genuine=0/49、max_abs_err=1.5e-2（fps 对拍）/3.5e1（flashfps 近似）——**全部一致**
- **门槛**（§14.9 vs task-6 §7.2）：8 条门槛判定值（含 per-frame 尾部 ti10 0.9425≥0.9246、modetest 0.8820≥0.8622、FPS<5ms 未达）——**全部一致**

## 4. 自检清单

- [x] §14 章节标题完整，11 小节编号连续、与正文 anchor 无断链
- [x] §14.8 四组实测表数据与 task-5/6 evidence 逐项一致（无凭空数字）
- [x] task-N evidence 引用全部可解析到实际文件，无悬空引用
- [x] §14.10 profile 隐患（max_n=6500 < 6988）有 task-6 §0 来源，非文档自造
- [x] 语义标注完整：§14.7 明确本实现对齐上游仓库（升序填充）而非论文（随机填充）
- [x] §14.11 实施记录 6 个问题与 task-1/4/4b evidence 描述一致

## 5. 结论
- plugin.md §14（v15.0）交付完整：三算法 FPS 插件族的设计、接线、语义论证、实测回填、门槛判定与遗留项齐备
- 全部实测数据可追溯到 task-{1..6}{,b} evidence，无悬空引用、无未经来源的数据
- F2/F1 终验通过所需文档证据齐备

## 6. 未做（遵守 MUST NOT DO）
- 未改动 plugin.md 内容（本文档为 task-7 的证据补记，非文档修改）
- 未改任何插件/kernel/onnx 代码（task-7 纯文档任务）
- 未做 git 操作
