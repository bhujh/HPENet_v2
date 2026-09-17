# issues — omitting-voxelize

## P2 发现（2026-09-16）
1. 【重要·阻塞 C7 构建】文档口径 build 在 clean 状态下**链接失败**（缺 libcudnn.so.8）：本机无系统 cuDNN，pip nvidia-cudnn-cu11 只有 `libcudnn.so.8` 无 `libcudnn.so`，CMakeLists.txt:76-85 的 find_library（ENV LD_LIBRARY_PATH）与 `${PYTHON_EXECUTABLE}` fallback（指向不存在的 libcudnn.so）均落空。本次基线以额外 `-DCUDNN_LIB=<pip 路径>/libcudnn.so.8` 构建成功；运行期还需 `LD_LIBRARY_PATH` 含该目录（否则连 `--help` 都 exit 127）。旧 build/ 目录是 CPP_trt3 拷来的（旧 link.txt 带 `-L.../cudnn/lib -Wl,-rpath-link` 掩盖了此问题；旧缓存 CUDNN_LIB=NOTFOUND）。**C7 若按文档原样 clean rebuild 会失败——是否修 CMakeLists 探测逻辑属方案外决定，暂只记录。**
2. 【事实与文档不符】`hpenet_v2_fp32.engine` 实际 14,679,492 B / 2026-09-07 20:42（AGENTS.md 的 14,446,844 B / 2026-08-26 已过时，属旧 `20260825-161134` pth）。当前 engine 与钉死 ckpt（`20260907-170521`）时间吻合；本次未验证其 pth 出处，C7/F4 对拍前建议确认。
3. 【小】`--help` 文案称 voxel_size 默认 0.02，代码实际 0.0001f（main.cpp:31）——help 文案过时，C5 删参时顺带留意。
4. 【小】`--output` / `--benchmark` 被解析但 CLI 路径不使用（output 仅 C-API 写 PLY）——C7 复测时勿依赖 `--output` 产出文件。
5. 【无害】编译警告 `cuda_utils.h:26 unused template param T`；运行期 TRT WARN「linked cuDNN 8.9.0 / loaded 8.7.0」（pip 包版本），对本次 fp32 引擎推理无影响。


## C7 发现（2026-09-16）
6. 【已查明·非缺陷·任务书前提修正】任务书 C7 称 CLI acc「期望逐位一致」（前提：新 C++ shuffle 复现 voxelizer.cu 的 RNG 流 → 精确复现）。实测 **非逐位一致**（mean Δ +0.14pp，逐文件 −1.14~+1.76pp）。根因：RNG 流确实一致（同 seed、同 N 长 Fisher-Yates，C++/numpy 交叉验证逐元素一致），但**被打乱的输入数组不同**——旧 voxelizer.cu 打乱的是 FNV-hash 排序点序 idx_s（伪随机排列），新代码打乱的是扫描序 iota。spec §3.3/§3.4 本来就声明「非 bit 等价 → 按 acc 验收」（Δacc<0.3pp），实测 0.14pp 在判据内；但后续 F4 若沿用以「逐位一致」为前提的断言需修正。C1-C6 代码与 spec §3.3 片段逐字吻合，无需改动。
7. 【小】clean rebuild 后二进制 1,596,440 B（旧 3,777,896 B 系 CPP_trt3 拷来的旧 build 产物），非异常，仅口径差异。
8. 【口径提醒】nsys 2022.4.2 不再随 profile 自动生成 .sqlite，analyze_latency.py 需要手动 `nsys export --type sqlite`（基线 sqlite 元数据证实当初同法）。

## B4 发现（2026-09-16）
9. 【重要·非 A/B 引入·阻塞 onnx_inference.py 端到端】`python deploy/onnx_inference.py --num_files 10`（默认 `--onnx deploy/hpenet_v2_plugin.onnx`）在 ORT 会话创建即 **exit 1**：`Fatal error: hpenet:FPSPrune(-1) is not a registered function/op`。该 ONNX（2026-09-07）含 5 个 domain 自定义算子（FPSPrune/PrefixFPS/BallQueryGroup/BallQueryDP/ThreeInterp），而 `onnx_inference.py` 与 `deploy/onnx_ops/*.py` 只有 torch 导出侧 symbolic 实现，仓库内不存在任何 onnxruntime 侧算子注册 → 该模型从未能在 ORT Python 加载。A3③（:269）与 B2 站点均在会话创建之后，此失败与 A/B 改动无关（先于 A3③ 行发生）。用无自定义算子的 `onnx_model_feat5_bn_sim.onnx`（Aug-13 老模型）跑通全脚本 exit 0、**无 TypeError**（A3③ 运行时得证），但 acc 无验收意义。F3 若要求 onnx_inference.py 做 acc 对拍，需先解决 ORT 自定义算子注册（含 CPU 语义实现；FPSPrune 的 prune 语义仅在 TRT 插件路径存在）——属方案外决定，暂只记录。

## 评审驳回记录（2026-09-16，独立评审 REJECT，append）

- 独立评审 **驳回** 本 plan 的验收结论，提出 **三项 blocker**：
  1. **部署口径延迟不达标 + Orin 未测**：部署口径实测 −1.6%（< 5% 判据），且未在 Orin AGX 上测过；端到端 −9.4% 被 PLY_LOAD（IO 噪声）主导（详见 learnings.md 更正节）。
  2. **部署 Python 判据未真正测量 + onnx runtime 证明取自无关模型**：`onnx_inference.py` 的 acc 从未在可加载的 plugin 模型上测得（ORT 自定义算子未注册，见本文件 #9）；所谓 ONNX runtime 证明用的是标准算子的 Aug-13 老模型，与目标模型无关，不能作为该行的验收证据。
  3. **C-API 去体素化前后从未对拍**：C-API 旧行为（voxel 0.02f 真体素化）无基线 acc 记录，旧二进制已被 clean rebuild 覆盖，未做前后直接对比（见 learnings.md C7 节 D 段局限声明）。
- **处理分工（lane 指派）**：
  - Blocker 1（延迟 + Orin）与 Blocker 2（部署 Python 口径）属测量工作 → 由**测量 lane**（需 GPU 空闲）承接。
  - Blocker 3（C-API 前后对拍）依赖 `deploy/CPP_trt3` 的 pristine 基线二进制 → 由**以 CPP_trt3 为 before 镜像的 lane** 承接。
  - 本 lane 只负责非测量类 should-fix（两处陈旧注释 + 本 notepad 的证据更正/卫生注记），不改动任何测量结论。
- 关联：验收状态与 checkbox 由 orchestrator 维护，本 entry 仅记录评审事实与分工，不修改 plan/状态文件。

## F4 发现（2026-09-16，append）

10. 【规格缺陷·需用户裁定·F4 确认】§5.2 部署（C-API）行的「与去体素化后 Python 参考 acc 之差 < 0.3pp」对 TRT/C-API 路径**不可直接适用**：C-API 唯一可跑的是 FPSPrune(keep_rate=0.75) TRT 引擎（变体 A），Python 参考是无 prune PyTorch 模型（变体 B），实测 Δ = 93.72 − 93.20 = **+0.52pp > 0.3pp**——但该差值是**模型变体差**：同变体对拍 C-API vs trt `_trt` = 0.0000pp、`_pytorch` vs Python 参考 = −0.21pp、去体素化本身 F2 = −0.07pp 均过线。**字面判据 FAIL，但根因是规格把两个不同模型变体放在同一行比较**。裁定（改判据为同变体对拍 / 或接受 0.52pp 为变体基准）属用户决策，F4 不静默通过。
11. 【残留缺口·记录在案】CPP_trt3 前后对拍（真实 C-API before）仍不可用（外部账户/额度限制），非本任务可解。
12. 【环境噪声·测量口径警告】本机 128 核 CPU 长期 load avg 100~150（GPU 6/7 的 vLLM 引擎挤占），宿主侧 NVTX 区间（SUBCLOUD_LOOP/TAIL）单次测量可被调度等待撑大 ±0.6ms（C7 单次 +0.943 的根因；f4_run4 的 PLY_LOAD 达 13.7ms 同源）。**任何单次 nsys 延迟结论在此机器上必须附多次复跑**。
13. 【小·工具行为】nsys 2022.4.2 的 `nsys export --type sqlite` 默认把 .sqlite 写到**当前工作目录**（非 .nsys-rep 同目录）；本轮首轮 export 误在仓库根生成 5 个文件（已删除）。后续 export 务必在目标目录执行或显式指定输出。
14. 【结论·F4】部署口径 ≥5%：6 次改动后运行 mean −10.2%（仅 C7 单次 −1.6% 未达）→ 均值达标；端到端 −15.4%（去 IO 离群后）但被 PLY_LOAD 主导，不作为收益口径。Orin（192.168.137.40）本机不可达（无路由、ping 丢包）。

## F4b 发现（2026-09-16，append）

15. 【无新异常·噪声定量补充】成对重复测量（8 运行）把 issues #12 的「单次 NVTX 宿主区间可被调度等待撑大 ±0.6ms」定量收紧：同一文件跨 8 次运行 SUBCLOUD_LOOP 极差 1.3~2.8 ms（最高单值 4.562 ms 出现在 after 侧）→ **任何单次 SUB/TAIL 对比在此机器上均无判别力**；部署口径（SUB+TAIL+VOX+COORD+ARG）才是噪声稳定的口径（B 4.014~4.939 vs A 3.576~3.947，完全分离）。后续延迟判定必须用部署口径 + 成对重复。
16. 【小·确认】PLY_LOAD 在本 lane 的 B1/B5 再次出现 13~14 ms（页缓存噪声，issues #12 同源）→ 端到端口径依然不可用于收益判定，仅作记录。
