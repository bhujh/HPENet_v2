## [2026-08-19] Task 1+5 完成
- **task() 基础设施**：category=...（Sisyphus-Junior）尝试用 `opencode/gpt-5-nano` 模型（不存在）→ 全部失败。必须用 `subagent_type="general"`（deepseek-v4-flash）才能派发实现任务。explore/oracle/momus 走 subagent_type 正常。
- **task 1 SampleFPS kernel 完成**：正确性全过（两级判据 100%、fuzz 8/8、N=10000、B=3、退化不死锁）。但 **perf N=5500/M=2750 = 70.74ms vs 现役 17.93ms（慢 ~4×）**！桶结构在 B=1 小 N 下开销主导（~8.2K 桶操作/轮 prune+reduce）。这是 plan 预期之外的重要发现——SampleFPS 精确模式在目标画像上不具提速价值，task 6 验收时"FPS<5ms"目标不能靠 SampleFPS 精确模式达成（Plan task 6 已注明"未达则记录瓶颈分析而非硬凑"）。FlashFPS prune+Cache 仍是唯一可能达标路径。
- **task 5 脚本就绪**：deploy/tests_fps_algos.py 一键跑通，自检 PASS，现役 fps 16 组 100%。三坑已规避。deferred acceptance 契约（NOT_READY 明确报错）验证通过。真实帧加载用 data/RadarClassi/radarfullwl/raw PLY。
## [2026-08-19] Task 2+4 完成
- task 2 SampleFPS 插件：照 fps_plugin.cpp 骨架，仅 stride 属性，nm 符号确认，tests_fps_algos samplefps 100%
- task 4 FlashFPS 三件套：独立类 flashfps_plugin.{cpp,h}（不共类）、flashfps_op.py（hpenet::FlashFPS, stride_i+keep_rate_f）、prune 路径在 samplefps_kernel.cu（N_active 建桶 + num_points 轮 + 升序填充位图 scan）
- **prune 语义验证脚本 deploy/tests_flashfps_prune.py**（仓库内持久化，权威判据）：
  - prefix_exact（前 num_points vs furthest_point_sample 逐索引）+ tail_ascending（尾部升序未选中）+ verify_fps_prefix 100%
  - 退化 k=0.05 → full_M_perm；k=0.001 → idx==arange(M)；确定性 OK
- **tests_fps_algos.py 对 flashfps_k075 报低一致率是预期的**（harness 用全 N 精确 FPS 判据，prune 语义本就不同）——不要误判为 bug，判读以 tests_flashfps_prune.py 为准
- 性能信息：N=5500 k=0.75 18.9ms vs 精确 30.7ms（候选剪枝生效，但现役仍 11ms）
- **脚本运行注意**：tests_flashfps_prune.py 全跑需 ~8 分钟（多档 engine build + 真实帧），用 `python -u` 关缓冲查看进度；200s timeout 不够
## [2026-08-19] Task 3+4收尾 完成
- task 3：samplefps_op.py + PrefixFPS 插件（workspace=0, kFLOOR_DIV 动态形状, arange fill kernel）+ onnx_backend fps_algo 开关（fps 默认回归 628 节点逐节点一致）+ onnx_export --fps-algo。PrefixFPS 动态 N=1024/6500 运行时断言过。
- task 4 收尾：flashfps 档 wiring——模块名末两段 <stage>.<subidx> 判据（stage==1→FlashFPS, stage>=2→PrefixFPS），实测 Patched 1+3。flashfps 图 1×FlashFPS(0.75)+3×PrefixFPS，TRT parse/build/deserialize/e2e 全过。
- **prefixfps_op.py 必要修复**：forward 由 1-D arange 改 (B,M)（否则全模型 trace 时 expand 报错；(B,M) 才是插件 getOutputDimensions 约定）。此修复是 task 3 产物但 task 4 收尾时才发现，已记录。
- **cache-only 语义实证**：flashfps k=1.0 图第 2-4 级输出 == samplefps 图对应级 == arange(M_ℓ)（N=4096/5500 双档），前缀等价性直接证明。注意验证是在 torch 侧 patch 模型上做的（ORT 无法执行 hpenet 算子），TRT 侧等价性由 task 1/2/5 的逐索引一致性外推。
- keep_rate·M₁ ≥ M₂ 约束：N=5500 时 2062 ≥ 1375 满足。
- 导出图产物：fps_algo_fps.onnx / fps_algo_samplefps.onnx / fps_algo_flashfps.onnx（628 节点）+ flashfps engine（15.3MB）。
## [2026-08-19] Task 6 完成 — 三算法端到端对比
- **acc**：samplefps & cache-only == 现役 0.9741（ti10 逐文件一致）；flashfps k0.75 ti10=0.9707（差门槛 0.0005）但 modetest=0.9338 反超现役 +0.0017；档位阶梯 k1.0/0.75/0.5/0.25 = 0.9741/0.9707/0.9612/0.9535（单调）
- **logits**：samplefps vs fps 逐位差 1.5e-2 全平局归因（idx mismatch 0.0021%，seq_exact 100%，0 genuine）；flashfps 近似（max_abs 3.5e1, pred flip 1.1-2.6%/frame）
- **延迟（同一 13 subcloud 工作负载 nsys）**：FPS 段 fps 106ms → samplefps 1080ms → flashfps 299ms(-72%)；端到端 median fps 28.8 / flashfps 43.7 / samplefps 134.5ms
- **FPS<5ms 目标未达**：瓶颈=samplefps_iterate_kernel 单 block 结构（gridX=1），level1-2 各 33-46ms/launch；现役 fps kernel 0.37-7.4ms/launch。多 block 并行化是后续 kernel 优化任务
- **⚠️ 重大部署隐患**：现役 v14 engine profile max_n=6500 < 测试集最大子云 6988 → 部分文件越界 acc 崩到 0.40！modetest 全量对比必须以新 fps engine（同 profile）为基线。这是现役部署 bug，需上报用户（独立于本计划）
- fps fp32 vs v14 非逐位（1.5e-2）是 profile 差异（opt5500 vs opt4096）→ conv tactic 选择不同，FPS 索引本身逐位一致
- flashfps 端到端未 ≤26.53ms（43.7ms）——同样因单 block 结构
## [2026-08-19] 终验波完成 — 计划全部闭环 (11/11)
- **F1 合规审计** PASS-WITH-NOTES：7 todo acceptance 全有 evidence；Must NOT 六项零违反；唯一缺口=task-7 evidence 缺失（已补写闭合）
- **F2 代码质量** APPROVE-WITH-FIXES：唯一 MAJOR=prefixfps_kernel.cu 越界写（缺 `i<total` 守卫，total 非 256 倍数时 OOB）——已修复（kernel 加 total 参数+守卫）+ 独立验证（N=2750/M=687 触发场景 PASS）；另补 static_assert 锁定 lsum[4]。无 thrust/参考仓库拷贝（KDLineTree 仅注释 attribution）
- **F3 真实 QA** PASS-WITH-NOTES：三 engine 完整推理 exit 0（acc fps/samplefps 0.9721 一致、flashfps 0.9679 符合预期）、tests_fps_algos 全绿、tests_flashfps_prune ALL PASS、nsys 抽查 FPS kernel ~3.8ms/subcloud（evidence 同量级）。NOTE：**同进程多 engine 共存段错误**（fps+samplefps+flashfps 三 TRTSession 同进程时 flashfps 段错误；独立进程全稳定）——疑似插件多 engine 状态冲突，记录待查，不阻塞单 engine 部署
- **F4 范围忠实度** PASS：无重训（.pth mtime ≤08-13）、openpoints/ 零 diff、现役 FPS/ballquery/threeinterp 零 diff、无 git commit、B 运行时参数、无交叉编译、新文件与 Scope IN 全对齐
- **Boulder**：fps-samplefps-flashfps-00000001 标记 completed
## [2026-08-19] fps_cache 收尾 + kernel 多 block 方案放弃
- fps_cache 档：已补 plugin.md §14.12（v15.1），全量 339 文件实测 acc 0.9569 四配置一致、延迟 -24~25%，Oracle 复审 APPROVE + 加固（else→elif stage白名单+raise）已落地
- **fps-kernel-multiblock 方案放弃**：双审 APPROVE-WITH-FIXES 后，数据核算证明多 block 必亏——现役 6.5ms/3600轮=1.8µs/轮，多 block 每轮 +2次 grid.sync(~1µs) → 3600×2×1µs≈7.2ms 已超 6.5ms。根因=贪心串行 M 轮 + 小 N 高采样比例（与 QuickFPS 同一反面画像）。方案标注 CANCELLED，未执行。
- **裁决记录**：Momus 误报「grid.sync 需 -rdc=true 否则编译失败」，Oracle 正确——实测最小 cooperative 探针默认 nvcc 编译运行全过，无需 -rdc。
- **最终交付格局**：fps（回退）/ samplefps（已证不适配）/ flashfps（实验）/ fps_cache（**生产推荐，零损失 -24%**）。瓶颈已从采样调度转移到单 kernel 串行结构，进一步提速需换采样范式或降 M，非单 kernel 并行度能解。
