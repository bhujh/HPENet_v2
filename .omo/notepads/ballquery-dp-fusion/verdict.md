# 融合 1（ball_query + dp kernel 融合）否决结论归档

> 归档日期：2026-08-23 | 状态：**否决，不落地**

## 结论

融合 1（把 `bq_dp_kernel` 融入邻居搜索 kernel）**净收益为负（−0.49%），否决，不落地**。`HPENET_BQ_DP_FUSION` 宏保持默认 0（现役路径），融合 .so 留作对照档。

## 背景与目标

- 目标：消除 `bq_dp_kernel`（占 GPU kernel 5.4%）的独立 launch + 对 idx/xyz 的重复读取，把 dp 计算融合进 `ball_query_kernel_fast`。
- 实现：`ballquerygroup_kernel.cu` 新增 `ball_query_dp_kernel_fast`，宏 `HPENET_BQ_DP_FUSION` 在融合/现役两路径间切换（默认 0），现役 `bq_dp_kernel` 原样保留作回退。

## 实测结果（V3 nsys A/B，stride-2 锚点 engine，真实文件 0000068.ply，iters=30）

| 量 | 数值 | 占比 |
|---|---|---|
| T_bq（现役 ball_query_kernel_fast） | 208,970,493 ns | 24.57% |
| T_dp（现役 bq_dp_kernel） | 41,663,711 ns | 4.90% |
| T_fused（融合 ball_query_dp_kernel_fast） | 254,818,429 ns | — |
| **净收益 = (T_bq+T_dp) − T_fused** | **−4,184,225 ns（−0.49%）** | **负 → 否决** |

## 否决原因（根因）

- **dp 写增量** = T_fused − T_bq = 45,847,936 ns = **1.10 × T_dp**，远超 0.5×T_dp 阈值。
- 融合 kernel 比纯搜索慢 **8%（grid=7）到 75%（grid=1）**。
- 根因：dp 是 channel-first `(B,3,M,S)` 布局，融合后每个邻居要写 3 段 `stride=m·nsample` 的**非连续（strided）写**，cache 不友好，把省下的 `bq_dp` 全部吃掉还倒贴 0.49%。

## 验证完整性（正确性无问题，纯粹性能负收益）

- T2 单元对拍：**69/69 bit 级一致** + 空球 compute-sanitizer 0 errors。
- T4 整网 E2E：4 组 engine（fp32/fp16 × 两档）**逐点 100% 一致**（0/41868 mismatch）。
- F1/F2/F3 终审：三个独立 reviewer 全 APPROVE（7/7、8/8、6/6）。
- 即：融合 kernel **语义等价无 bug**，只是 strided 写导致的性能倒挂。

## 后续方向

- dp 写增量 1.10×T_dp > 0.5×T_dp 阈值 → 触发**融合 1.5（dp 布局交错化 float3）**立项条件，但**暂缓**：收益仅 ~4% GPU kernel，且需同步改下游 HPE 消费方式（否则把 strided 写转移到 HPE 的读），高风险。

## 已关闭的延迟优化路线（汇总，勿重试）

1. **GridBallQuery**：搜索段慢 8.9~13.5×（plugin.md §15）
2. **融合 1 ball_query+dp**：净收益 −0.49%（本文档）
3. **FPS 多 block**：grid.sync 每轮开销（1.94µs）> 每轮计算（1.81µs），结构性必亏
