## [2026-08-19] SampleFPS 性能风险
- SampleFPS kernel 在 N=5500/M=2750 实测 70.74ms，比现役 fps_launcher (17.93ms) 慢 ~4×。根因：gridDim=B 单 block 下每轮全量扫 4096 桶做 prune+reduce（~8.2K 桶操作/轮），2750 轮叠加。
- 影响：task 6 的"FPS 合计 <5ms"目标**不可能**由 SampleFPS 精确模式达成；FlashFPS prune（keep_rate<1 减少轮数）+ Cache（第 2-4 级 PrefixFPS 归零）是唯一可达标路径。
- 决策点：task 6 验收时须按 plan 的"未达则记录瓶颈分析而非硬凑"处理，或在 SampleFPS 插件侧做多 block 优化（属计划外扩张，不建议首版做）。
