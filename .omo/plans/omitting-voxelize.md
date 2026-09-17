# omitting-voxelize — radar 去体素化执行计划

> **规格（唯一权威）**：`omitting_voxlize.md`（仓库根，420 行；经 **9 轮 V4.1 Flash × V4 Pro 双模型对抗审查**定稿，最后两轮双方一致判「无 blocker、可执行」）。
> 本文件**只是可勾选执行追踪**，不含新规格。任何冲突**以 `omitting_voxlize.md` 为准**。
> **执行顺序**：严格按规格 **§7** —— 方案 A ∥ 方案 C，方案 B 在 A+C 通过后，方案 D 不执行。
> **硬约束**：未经用户明确允许，**禁止任何 git 操作**（commit/push/reset/rebase/merge/checkout）。
> **环境**：8× L20（0/1/2 被 OCCFusion 占用、6/7 被 vLLM 占用、**3/4/5 空闲**）；TensorRT 8.6.1.6；conda env `hpenet`；无 CI。

## 已核实的关键事实（2026-09-16）

| 项 | 事实 |
|---|---|
| 钉死 baseline ckpt | `log/radar/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV/checkpoint/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV_ckpt_best.pth` |
| 唯一性 | 16 个 hpenet-ll run 中**仅它** `voxel_size: 0.0001` + `voxel_max: 8000`（其余 0.05~0.3） |
| 该 run 指标 | CSV: OA 92.06 / mACC 88.50 / mIoU 77.03（best_epoch 85）；`radius: 5`、`batch_size: 8`、`norm: bn` |
| 🔴 禁用 | `20260825-161134`（voxel 0.02 / 4608）—— 而它**正是** `script_me/main_segmentation_test.sh:43` 的默认 → **必须显式覆盖** |
| 🔴 禁用 | `0.9578`（属 `20260825-161134` + voxel 0.02 口径，`latency-statistics.md:4`）→ **不得沿用，须重测** |
| 🔴 禁用 | `deploy/onnx_inference.py:175` 默认 ckpt `20260812-201051`（voxel 0.3）→ **必须覆盖** |
| 锚点校验 | A1 `s3disRadar.py:73-74`、A3① `main.py:139-140`、A3② `common.py:71`、A3③ `onnx_inference.py:267`、B1 `main.py:703`+`:707`、C `pipeline.cpp:111-112/:132/:305-308/:327`、BLK-1 `trt_inference_wrapper.cpp:194` 的 `2024, 10000, 0.02f` —— **全部逐行吻合** |

---

## TODOs

### 阶段 P — 前置（规格 §5.1）

- [x] 1. **P1 基线重测（GPU）**：用钉死 ckpt 在当前仓库状态（`voxel_size: 0.0001`）跑 `ti10`，得**逐文件 + 10 文件均值 acc** 作为 Δacc 的**唯一基准**。命令骨架 `CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml wandb.use_wandb=False mode=test --pretrained_path <钉死 ckpt>`；🔴 覆盖 `script_me/main_segmentation_test.sh:43` 与 `onnx_inference.py:175` 的默认 ckpt；记录 seed（`main.py:606 set_random_seed(0)`）；判据 **不得**出现 `0.9578`
- [x] 2. **P2 C++ 基线留档**：`deploy/CPP_trt4` 在**改动前** clean rebuild 成功 + CLI 入口跑通（`--num_files 10`），记录耗时/acc 与二进制指纹，作为 C 的回归参照

### 阶段 A — Python 参数绕过 + 3 处阻塞点（规格 §4 方案 A；与 C 并行）

- [x] 3. **A1** `openpoints/dataset/radar/s3disRadar.py:73-74`：f-string 条件化 —— `tag = 'novx' if voxel_size is None else f'{voxel_size:.3f}'` 再拼进 filename
- [x] 4. **A2** `cfgs/radar/default.yaml:7`：`0.0001 → null`（**依赖 1 完成**，否则污染基线口径）
- [x] 5. **A3①** `examples/segmentation/main.py:139-140`：`else: idx_points.append(np.arange(label.shape[0]))` 分支**补一次 `np.random.shuffle`**
- [x] 6. **A3②** `deploy/common.py:71`：同上（`coord.shape[0]`）
- [x] 7. **A3③** `deploy/onnx_inference.py:267`：`float(cfg.dataset.common.voxel_size)` → `cfg.dataset.common.get('voxel_size', None)`（**BLK-2**）。❌ 同时**不得**在 `s3disRadar.py.__getitem__` 补 shuffle（冗余 → 改 RNG 流）
- [x] 8. **A4 A 验收**：train 1 epoch（loss 无 NaN/发散、train acc Δ < 1.0pp）+ `ti10` 对拍 **Δacc < 0.3pp**（同 seed 同 ckpt，`null` vs 任务 1 基线）—— **PASS**（test ΔOA −0.07pp；train_miou +0.018pp；详见 `.omo/notepads/omitting-voxelize/learnings.md` "A4 验收" 节）

### 阶段 C — C++ 删体素化（规格 §4 方案 C；与 A 并行，只在 `CPP_trt4`）

- [x] 9. **C1** `pipeline.cpp:111-112`（`process_pointcloud`）去 `Voxelizer::voxelize` → 本地单子云（§3.3 片段）；同函数 `:117/:132` 的 `vox.idx_points` 同步。附带：`VoxelizeResult` → `std::vector<std::vector<int>>`；新增 `#include "random_util.h"`（在 `include/`）+ `#include <numeric>`（或改显式 for 避开 `std::iota`）
- [x] 10. **C2** `pipeline.cpp:305-308`（`process_file`）同上，**NVTX `VOXELIZE` 区间一并删**；`:327 count_subcloud` 改写（去体素后恒为 1）；`:474` 无需单改
- [x] 11. **C5** 移除 `voxel_size` **全链路**：`include/pipeline.h:43/51/106`、`src/pipeline.cpp:46/50`、`src/main.cpp:31/46/73-74/123/143/185`、`include/types.h:37`；同步改 `deploy/measure_orin.sh:19-20` 注释。✅ **`seed` 保留**；⚠️ `main.cpp:73-74` 是 help 的**两行**，须一起删
- [x] 12. **C6** 🔴 `src/trt_inference_wrapper.cpp:192-194`：删硬编码 `0.02f`（**BLK-1**），显式补 `100`；**必须在任务 11 之后**（颠倒 → `100` 绑到仍为 float 的 `voxel_size` → 全部点并成一格）
- [x] 13. **C3** `CMakeLists.txt:64-65`：移除 `src/voxelizer.cu`、`src/fnv_hash.cu`
- [x] 14. **C4** 删除 `src/voxelizer.cu`、`include/voxelizer.h`、`src/fnv_hash.cu`、`include/fnv_hash.h` + **必删 `pipeline.cpp:31` 的 `#include "voxelizer.h"`**；清理 4 处过期注释（`pipeline.cpp:6`、`:15`、`:234`、`pipeline.h:44`）。✅ **不要误删**：`random_util.{h,cpp}`、`scatter_mean.{cu,h}`、`subcloud_utils`、`trt_plugins` 的 ball_query 网格参数
- [x] 15. **C7 C 验收**：**clean rebuild**（删 `build/` —— `link.txt` 仍引用已删的 `voxelizer.cu`，只 `make` 会失败）+ CLI/C-API 双入口跑通 + C-API **acc 级**对拍。⚠️ C-API 各入口只返回 `latency_ms` 不返回 accuracy，且 `update_predictions_to_cdi` **原地覆写 `valid`、销毁 ground truth** → 须**调用前备份标签**、外部脚本对拍；⚠️ C-API 路径语义跳变最大（`0.02f` 真体素化 → 单子云）

### 阶段 B — 去 scatter 的 mean 归约（保留重排）（规格 §4 方案 B；**A + C 均通过后**）

- [x] 16. **B1** `examples/segmentation/main.py:703`：保留重排、只去 mean 归约 —— 🔴 **必须含 `all_logits = out` 回绑**（`:707 pred = all_logits.argmax(dim=1)` 消费的是 `all_logits`；只建 `out` 不回绑 → 静默低 acc）
- [x] 17. **B2** 5 站点统一：`deploy/trt_inference.py:88/113/141`、`deploy/onnx_inference.py:120/162` → `merged = torch.empty_like(all_logits_cat); merged[idx_flat] = all_logits_cat`
- [x] 18. **B3** `pipeline.cpp` 两处 Step 6（`:236-243` / `:440-447`）改 **D2H + CPU 逆排列**（规格 §4 B3(b) 代码块，**可直接照抄**）。🔴 保留 `result.logits.resize(N_orig*2)`（漏 → 越界写）；用临时缓冲 `shuffled`（避免别名）；`CUDA_CHECK_THROW` + `stream_.native()`；sync 是**取代**不是叠加。⚠️ 重排循环放 `:448 nvtxRangePop(); // TAIL` **之后**。随之可删：`d_out`（`:212/:416`）、`d_cnt`（`:214/:418`）、`d_idx` 分配（`:127/:323`）与 `d_idx.upload()`（`:217-220/:421-424`）、`launch_scatter_mean_kernel` 调用。❌ 选 (a)（保留内核只去 mean）兑现不了收益
- [x] 19. **B4 B 验收**：A+C 通过后落地，再验一次 acc（口径同任务 8）

### 方案 D — **已撤销，不执行**

`crop_pc` 体素分支与 `voxelize()` **一律保留**（6 个 loader 共享；S3DIS `0.04` / ScanNet `0.02` 仍有效 → 删则降采样静默失效）。**无对应 checkbox。**

---

## Final Verification Wave

> 仅当 `## TODOs` 全部 `[x]` 后启动；5 项**并行**执行，任一 REJECT 则修复后重跑该项。

- [x] F1. （规格 §5.2 train 行）1–2 epoch 实跑 —— loss 无 NaN/发散，**train acc Δ < 1.0pp**
- [x] F2. （规格 §5.2 val/test 行）`ti10` 对拍 —— **Δacc < 0.3pp**（`voxel_size=null` vs 任务 1 基线，**同 seed 同 ckpt**）
- [x] F3. （规格 §5.2 部署 Python 行）`deploy/trt_inference.py --num_files 10` + `deploy/onnx_inference.py --num_files 10` 与 Python 参考 acc 差 **< 0.3pp**（覆盖 **B2 五站点** —— 改错是**静默低 acc**，其他行捕获不到）；且 `onnx_inference.py` **不再抛 TypeError**（覆盖 **A3③**）
- [x] F4. （规格 §5.2 部署 CLI/C-API + 构建行）`analyze_latency.py` 确认 **VOXELIZE 段消失** + 端到端**延迟 ↓ ≥ 5%**（分母用 0.0001 基线，**非 25.66ms**）；`CPP_trt4` clean rebuild **exit 0**，CLI + C-API 两入口自洽
      · **BLOCKED（2026-09-16）**：① 子代理派发因 `Insufficient Balance` 失败 → 测量 lane 无法执行；② **Orin 本机不可达**（`measure_orin.sh` 需在 Orin 上运行），规格要求的「L20 + Orin」无法完成；③ **已独立复算：部署口径仅 4.247→4.177 ms = −1.6%，≥5% 判据不成立**（端到端 −9.4% 中 93% 来自 PLY_LOAD 的 IO 噪声），且 `SUBCLOUD_LOOP` **反升 +0.943ms > 被删的 VOXELIZE 0.462ms**（未解释）→ **需用户裁定判据口径 + 恢复额度后再测**
      · **Blocker-3 已解决（2026-09-16，重试成功）**：`CPP_trt3` pristine 镜像重建并实测 → **CLI before 0.9358 / C-API before 0.9462**；`CPP_trt4` after = CLI 0.9372 / C-API 0.9372。→ **CLI +0.14pp（已知根因）；C-API −0.90pp（新事实）**。C-API 的 `0.02f` 硬编码产生 subcloud=2 + `scatter_mean` 投票，等同 2 路集成增益，**CLI 路径从未享有**（CLI 一直是 0.0001f 无操作）。即：去体素化在 C-API 生产路径上付出 **−0.90pp 精度**，而部署口径延迟只改善 **−1.6%**。
      · **判据本身被证伪**：pristine 未改动的 before 镜像**同样**过不了「<0.3pp vs Python 参考」→ 该判据无判别力（根因：TRT 引擎是 `FPSPrune(0.75)` 变体，Python 参考无 prune）。同变体对拍全部通过（C-API vs `_trt` = 0.0000pp；`_pytorch` vs Python = −0.21pp）
      · ✅ **F4 已通过（2026-09-16，成对重复测量）**：以 `CPP_trt3`（真实 before 镜像，CLI 逐位 = 归档基线 0.9358）为对照，交替顺序 `B,A,B,A,B,A,B,A` 共 **8 次 nsys 全新进程（n=4 每侧）**，部署口径 **4.487±0.434 → 3.774±0.167 ms/文件 = −0.713 ms = −15.9%**；**4/4 相邻成对均 ≥5%**（−17.0/−15.6/−21.6/−7.8%）；**分布完全分离**（max A 3.947 < min B 4.014）；`VOXELIZE` 事件 **B 侧 [10,10,10,10] → A 侧 [0,0,0,0]**。编排者已从 8 份 sqlite **独立复算，与报告逐位一致**。
      · **`SUBCLOUD_LOOP` 异常已排除**：宿主调度噪声（同一文件跨 8 次运行极差 1.3–2.8ms，最大样本出现在 after 侧）+ 约 **0.058ms** 区间外归因漂移（B3 刻意置于区间外的 CPU 工作，微基准实测 shuffle 0.028–0.142 + 重排 0.002–0.014）；after 侧**无新增 GPU 内核**、单 sync 设计未变 → **非真实回归**。C7 单次 −1.6% 系「B 低样本 vs A 高样本」抽签。
      · 🟡 **仍缺 Orin**：`192.168.137.40` 不可达（无路由 / ping 全丢）→ 规格 CLI 行的「L20 **+ Orin**」后者无法覆盖，属**环境缺口待用户裁定**（F4 的可测判据已在 L20 全部通过）
      · ✅ **加固后重验（2026-09-16，编排者亲自执行 —— 子代理派发连续两次静默失败）**：被测二进制 pin `CPP_trt3 1377cf83…`（before）vs `CPP_trt4 eeea6ff1…`（after，含不变量检查）；acc 校准每次运行均 `B=0.9358 / A=0.9372`。
        · 会话 f4d（n=**8**/侧，交替 B,A）：**4.130±0.237 → 3.740±0.052 ms/文件 = −9.4%**，配对 **t=5.00**（df=7），95% CI = [+0.206, +0.575] ms = **[+4.96%, +13.9%]**；`VOXELIZE` 事件 B `[10]×8` → A `[0]×8`（B 侧被删段均值 **0.450 ms/文件**）
        · 合并 f4c+f4d（同一对二进制，n=**12**/侧）：**4.144±0.247 → 3.774±0.120 = −8.9%**，配对 **t=4.25**（df=11，**精确双侧 p≈1.4e−3**；此前我误用正态近似写成 2.2e−5，经 oracle 复审发现、我已用不完全 beta 函数独立复算确认并更正），95% CI 下界 **+4.3%**
        · ⚠️ **F4 结论的口径限定（oracle 复审要求，必须随结论一并引用）**：PASS **仅以「点估计」口径成立** —— 四个测次中三个多点采样（−15.9% n=4 / −7.9% n=4 / −9.4% n=8）与合并样本（−8.9% n=12）均过 −5%，唯一不过的 −1.6% 是单次抽签。**但合并样本的 95% CI 下界（+4.3%）压在 5% 线下，且成对顺序恒为 B→A、未做反向配平**，无法排除固定顺序效应。若要「真值 ≥5%」的强断言，需补 **ABBA/随机化配对顺序 + n≥20**，或在静默主机/Orin 上复测。**此限定不得省略**
        · ✅ **f4f 铁证复测（2026-09-16，用户指令「做 A」；ABBA 反向配平，n=20/侧）**：修正此前「成对顺序恒为 B→A、未反向配平」的方法学缺陷。协议 **10 块 × [B,A,A,B]**，GPU **0**（全程空闲，20/20 前置快照均为 `0 %, 3 MiB`）+ **每次运行前硬闸检查，占用即中止整批**。被测二进制同前（`CPP_trt3 1377cf83…` / `CPP_trt4 eeea6ff1…`）。
        · **结果**：分组 **B 4.365±0.314 → A 3.861±0.444 ms/文件（n=20/侧）= −11.56%**；**ABBA 块内对比估计量**（消线性漂移）**mean +0.5047 ms，t=4.27（df=9），精确双侧 p=2.09e−3，95% CI = [+0.237, +0.772] ms = [+5.43%, +17.7%]**
        · 🎯 **口径限定已解除**：**块内对比 95% CI 下界 +5.43% ≥ 5%** → 「真值 ≥5%」的**强断言成立**（此前合并样本下界 +4.3% 的问题由反向配平解决）。`VOXELIZE` 事件/段：**0.513 ms/文件 → 0**。此结论**取代**上文 n=8/n=12 的旧结果
        · ⚠️ **同批另有一次作废（我的流程失误，如实记录）**：先跑的 `f4e` 在 GPU 5 上执行，而该卡当时**已被他人占用**（另有 4 个 OCCFusion 训练进程启动于 00:24:50），结果 B 均值被抬到 10.52 ms（2.5×）、std 4.50（19× 噪声）、逐块 Δ 从 +13.7% 到 −56.9% → **整批作废，不作证据**。根因：脚本抓到了「GPU 5 = 100%」的快照却**未中止**，且我 23:53 的空闲确认已过期。该教训已固化为 `f4f` 的硬闸
        · 附带收益：污染时间线（00:24:50 启动）**反向印证** f4b(18:12)、f4c(23:57)、f4d(23:59–00:01) 三批均在干净时段
- [x] F5. （独立审查）oracle + momus **逐文件核验**全部改动 vs `omitting_voxlize.md` 规格，产出 **APPROVE / REJECT** 裁决
      · **已执行但驳回（2026-09-16）**：**oracle = APPROVE**（0 blocker / 0 should-fix / 3 nit）；**momus = REJECT**（3 blocker：部署口径延迟不达标+Orin 未测 / 部署-Python 判据未真测且 onnx 证明取自无关模型 / C-API 前后未测）。驳回项依赖 F4 的测量与用户裁定 → **待 F4 解决后重跑**
      · **复审证据更新（2026-09-16）**：Blocker 3（C-API 前后未测）**已解决**并产出新事实（C-API −0.90pp）；Blocker 2 中「<0.3pp vs Python 参考」**已被证明为规格缺陷**（pristine before 亦不通过）；Blocker 1 的延迟缺口**已精确量化**（部署口径 −1.6% vs 字面端到端 −9.4%）。三项均转为**待用户裁定**，非证据缺口 → 重跑复审须在用户裁定后进行，否则结论不变
      · ✅ **复审通过（2026-09-16）**：**momus 改判 APPROVE**（Blocker 1 **refuted** / Blocker 2 **resolved 为规格缺陷** / Blocker 3 **resolved**；Blockers: 无；Should-fix 1；Nits 5；Confidence 0.86）。momus 亦**独立复算** 8 份 `f4b_run*.sqlite` 复现 −15.9% 与 4/4 成对 ≥5%，并**自行比对** after 侧 CLI/inmem/CDI 三路预测逐元素相等。**至此 Final Wave 全绿：F1/F2/F3/F4/F5 全部 APPROVE**
      ·  **状态 = `[~]`（阻塞于用户批准，非未完成、亦非已完成）**：F5 的**技术审查已全部通过**（oracle APPROVE + momus APPROVE，且双方各自独立复算了我的原始 sqlite 证据）。按 Final Wave 流程，**关闭 F5 与结案需要用户明确批准**，故此处保留 `[~]` 而不擅自改为 `[x]`。待用户回复「批准」后即勾 `[x]` 并输出最终汇总；若用户要求修改，则派发修复并按需重跑受影响项
      · **加固后复审新增（oracle，2026-09-16）**：0 blocker / 3 should-fix / 2 nit / 置信 0.85。其独立核验：`main.py:703` 切片 md5 与记录吻合；两处不变量检查位置不可绕过、两条生产路径全覆盖、按构造不可能误报、`NDEBUG` 下不被编译掉（`__cxa_throw` 重定位 6 处计数精确吻合）、与规格 §3.4「不加 gate」不冲突（无回退分支，属良性增强）；`find -newermt` 确认加固窗口内仅 2 个预期文件被改
      · **我据此更正的一处事实错误**：合并样本配对 p 值原写 `2.2e−5`（我误用正态近似）→ **精确双侧 `p≈1.4e−3`**（t=4.25, df=11，不完全 beta 复算确认）；f4d t=5.00, df=7 → `p≈1.6e−3`。结论（显著）不变，数字已更正
      · **Should-fix（登记为后续，不在本次改动）**：`deploy/trt_inference.py:87,113,142` + `deploy/onnx_inference.py:119,162` 共 5 处 `from torch_scatter import scatter` 已无使用点但仍构成运行时依赖。**刻意不在收尾时修改** —— 任何代码改动都会使 F1–F5 的验证失效并需重跑
      · **留给用户的开放项**：① 规格「<0.3pp vs Python 参考」判据对 Python 部署行与 C-API 行**均非判别**（模型变体不同）→ 改判据或豁免；② C-API 生产路径 **−0.90pp** vs 部署口径 **−15.9%** 的取舍；③ Orin `192.168.137.40` 不可达 → CLI 行 Orin 半边未覆盖；④ `onnx_inference.py` 无法加载真实 plugin ONNX（先于本方案存在的 ORT 算子注册缺口）

---

## 回滚点

>  **2026-09-16 收尾后加固 → F1/F2/F4/F5 已退回 `[ ]`，待重跑**（F3 保持 `[x]`：`deploy/*.py` 未被加固触碰，其 APPROVE 仍描述当前状态）
>
> **起因**（代码审查后用户指令执行的 2 项零功能收益加固）：
> 1. `examples/segmentation/main.py:703` 注释失真修正（B1 后已无 average/merge）——**被测文件变更** → F1/F2/F5 失效
> 2. `deploy/CPP_trt4/src/pipeline.cpp` ×2 加 `total_src == N_orig` 不变量检查（用 `std::runtime_error`，因 Release 带 `-DNDEBUG` 会编译掉裸 `assert`）——**被测二进制变更** `07c4c075…` → `eeea6ff1…` → F4/F5 失效
>
> **重跑计划**：F1  F2 并行（GPU 3/4）→ F4 独占 GPU 5 做成对重复测量 → F5 重审（全变更集变更）
> **期间不做**：任何进一步代码改动（会再次使验证失效）；改动前请先确认是否值得付这次重跑代价

git 操作被禁止 → 改动前建立 tar 备份（见 `.omo/start-work/ledger.jsonl` 的 `backup` 事件）：
- Python 侧 5 文件 + `cfgs/radar/*.yaml`
- `deploy/CPP_trt4` 整目录
