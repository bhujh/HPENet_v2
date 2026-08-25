# 融合 1：ball_query + dp kernel 融合 实施计划

## 目标
消除 `bq_dp_kernel`（现占 GPU kernel 5.4%）的独立 launch 及其对 `idx`/`xyz` 的重复读取，
把 dp 计算融合进邻居搜索 kernel。预期整网 GPU kernel 时间下降约 4–5%，精度零回退（bit 级）。

## 现状（源码已确认）

`BallQueryGroup`（ballquerygroup_plugin.cpp）I/O：
- 输入 0=xyz(B,N,3) fp32、1=new_xyz(B,M,3) fp32、2=features(B,C,N) fp32|fp16
- 输出 0=grouped(B,C,M,S)、1=dp(B,3,M,S) fp32
- workspace idx_ws(B,M,S) int32

`BallQueryDP` I/O：
- 输入 0=xyz、1=new_xyz
- 输出 0=dp(B,3,M,S) fp32、1=idx(B,M,S) int32（idx 直接写输出 tensor，无 workspace）

launcher 内部（ballquerygroup_kernel.cu）：
```
ballquerygroup_launcher:  ball_query(写 idx_ws) → bq_dp(读 idx_ws→写 dp) → bq_gather(读 idx_ws→写 grouped)
ballquerydp_launcher:     ball_query(写 idx 输出) → bq_dp(读 idx→写 dp)
```

`ball_query_launcher_with_stream` 调用方（3 处）：
- ballquery_plugin.cpp:68（旧 BallQuery 插件，仅需 idx，**必须保持签名不变**）
- ballquerygroup_kernel.cu:75（group）
- ballquerygroup_kernel.cu:106（dp）

`bq_dp_kernel`（ballquerygroup_kernel.cu:17）语义：
- dp = (xyz[k] − query) / radius（normalize_dp=1）或 (xyz[k] − query)（normalize_dp=0）
- 写布局 dp[(bs*3+c)*m*nsample + pt*nsample + s]，c∈{0,1,2}
- 遍历 idx[0..nsample-1]（含首邻居 padding 的重复槽位）

## 方案：新增融合 kernel，不动现役 ball_query_kernel_fast

### 1. 新增 `ball_query_dp_kernel_fast`（放 ballquerygroup_kernel.cu，匿名 namespace）

（`THREADS_PER_BLOCK` 来自 `cuda_utils.h`，nsys 实测 BlockXYZ=256，与现役 `ball_query_kernel_fast` 一致）
（融合 kernel 的 `__global__` 定义包进 `#if HPENET_BQ_DP_FUSION`，宏 0 时完全剔除，保证 A/B 两个 .so 纯净）

签名：
```cuda
__global__ void ball_query_dp_kernel_fast(
    int b, int n, int m, float radius, int nsample,
    float inv_radius, int normalize_dp,
    const float* __restrict__ new_xyz, const float* __restrict__ xyz,
    int* __restrict__ idx, float* __restrict__ dp);
```

循环体（严格复制 ballquery_kernel.cu:20-36 的邻居搜索语义，§13.8 禁止手写改写；只加 dp 写）：
```cuda
int bs_idx = blockIdx.y;
int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
if (bs_idx >= b || pt_idx >= m) return;
new_xyz += bs_idx*m*3 + pt_idx*3;
xyz     += bs_idx*n*3;
idx     += bs_idx*m*nsample + pt_idx*nsample;
dp      += bs_idx*3*m*nsample + pt_idx*nsample;   // dp 基址 = c0 段起点
float radius2 = radius * radius;
float new_x = new_xyz[0], new_y = new_xyz[1], new_z = new_xyz[2];
float inv_r = normalize_dp ? inv_radius : 1.0f;
int cnt = 0;
for (int k = 0; k < n; ++k) {
    float x = xyz[k*3+0], y = xyz[k*3+1], z = xyz[k*3+2];
    float dx = x - new_x, dy = y - new_y, dz = z - new_z;  // dp 语义 = xyz-query
    float d2 = dx*dx + dy*dy + dz*dz;                       // 平方，符号无关，与现役 d2 值一致
    if (d2 < radius2) {                                     // 严格 <，与现役一致
        dx *= inv_r; dy *= inv_r; dz *= inv_r;
        if (cnt == 0) {                                     // 首邻居 padding：idx+dp 同时填充
            for (int l = 0; l < nsample; ++l) {
                idx[l] = k;
                dp[0*m*nsample + l] = dx;
                dp[1*m*nsample + l] = dy;
                dp[2*m*nsample + l] = dz;
            }
        }
        idx[cnt] = k;
        dp[0*m*nsample + cnt] = dx;
        dp[1*m*nsample + cnt] = dy;
        dp[2*m*nsample + cnt] = dz;
        ++cnt;
        if (cnt >= nsample) break;                          // early break 不变
    }
}
```

### 2. 修改两个 launcher（签名不变，用编译期宏在融合/现役两路径间切换）

在 ballquerygroup_kernel.cu 顶部定义开关（默认关闭=现役，A/B 测试时开启）：
```cuda
// 0 = 现役路径(ball_query + bq_dp)，1 = 融合路径(ball_query_dp_kernel_fast)
#ifndef HPENET_BQ_DP_FUSION
#define HPENET_BQ_DP_FUSION 0
#endif
```

`ballquerygroup_launcher`：
```cuda
float inv_radius = (radius != 0.0f) ? 1.0f/radius : 1.0f;
dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);
dim3 threads(THREADS_PER_BLOCK);
#if HPENET_BQ_DP_FUSION
    ball_query_dp_kernel_fast<<<blocks, threads, 0, stream>>>(
        b, n, m, radius, nsample, inv_radius, normalize_dp, new_xyz, xyz, idx_ws, dp);
#else
    ball_query_launcher_with_stream(b, n, m, radius, nsample, new_xyz, xyz, idx_ws, stream);
    bq_dp_kernel<<<blocks, threads, 0, stream>>>(
        b, n, m, nsample, normalize_dp, inv_radius, xyz, new_xyz, idx_ws, dp);
#endif
// 随后原样保留 bq_gather_kernel（读 idx_ws → 写 grouped），两路径共用
```

`ballquerydp_launcher`：同上，idx/dp 均为输出 tensor：
```cuda
#if HPENET_BQ_DP_FUSION
    ball_query_dp_kernel_fast<<<blocks, threads, 0, stream>>>(
        b, n, m, radius, nsample, inv_radius, normalize_dp, new_xyz, xyz, idx, dp);
#else
    ball_query_launcher_with_stream(b, n, m, radius, nsample, new_xyz, xyz, idx, stream);
    bq_dp_kernel<<<blocks, threads, 0, stream>>>(
        b, n, m, nsample, normalize_dp, inv_radius, xyz, new_xyz, idx, dp);
#endif
```

### 3. 保留 `bq_dp_kernel` 不动（作为现役回退/对照路径）
- 不删除 `bq_dp_kernel`（ballquerygroup_kernel.cu:17-45 原样保留）
- 它仍是 `HPENET_BQ_DP_FUSION=0` 时的现役路径，A/B 测试与回退都依赖它
- 融合 kernel 是新增的**平行实现**，通过宏切换，与项目"档位保留"哲学一致

### 4. 保留 include
`#include "ballquery_kernel.h"` 必须保留（现役路径仍调用 `ball_query_launcher_with_stream`）。

## 语义一致性要点（必须逐条核对）

| 要点 | 现役 | 融合后 | 结论 |
|---|---|---|---|
| dp 符号 | xyz[k]−query | x−new_x（同） | 一致 |
| d2 判据 | (new_x−x)² 严格 < r² | (x−new_x)² 严格 < r² | 平方等价，一致 |
| normalize_dp | ×inv_radius 或 ×1 | 同 | 一致 |
| 首邻居 padding | idx[0..S-1]=k | idx+dp 同填 | 一致（dp padding 槽位 = 首邻居 dp，对应现役 bq_dp 重复算首邻居） |
| early break | cnt≥S 停 | 同 | 一致 |
| 空球 cnt==0 | idx 未初始化（§15.7 已知 bug），bq_dp 读垃圾 idx | idx+dp 均不写（未初始化），但**不再越界读 xyz** | 融合后更安全，不引入新 UB；不主动加 memset（避免范围膨胀） |
| dp 写布局 | (B,3,M,S) strided | 同 | bit 级可对拍 |

## 改动文件清单

1. `deploy/trt_plugins/src/ballquerygroup_kernel.cu`
   - 顶部加 `HPENET_BQ_DP_FUSION` 宏（默认 0）
   - 新增 `ball_query_dp_kernel_fast`
   - 修改 `ballquerygroup_launcher` / `ballquerydp_launcher`（宏切换融合/现役两路径）
   - **保留** `bq_dp_kernel`（现役路径）、保留 `#include "ballquery_kernel.h"`
2. 无 header 改动（launcher 签名不变）
3. 新增 `deploy/trt_plugins/tests/test_bq_dp_fusion.cu`（V1 独立 nvcc 对拍程序）

## 验证计划（TDD：先对拍后性能）

### V1 单元对拍（新建 `deploy/trt_plugins/tests/test_bq_dp_fusion.cu`，独立 nvcc 程序）

- **对拍机制**：单一 CUDA 程序内 `#include "../src/ballquery_kernel.cu"` 与 `#include "../src/ballquerygroup_kernel.cu"`（CUDA 支持 include .cu；注意 include 顺序与 guard 避免重定义，或把对拍程序放 src/ 下单独 nvcc 编译），对同一输入分别跑「现役路径（`ball_query_kernel_fast` + `bq_dp_kernel`）」与「融合路径（`ball_query_dp_kernel_fast`）」，对比 dp/idx 输出——**不经 TRT/插件层**。
- **编译宏要求**：对拍程序编译必须加 `-DHPENET_BQ_DP_FUSION=1`——融合 kernel 定义包在 `#if HPENET_BQ_DP_FUSION` 内（宏 0 时剔除），不加此宏则对拍程序拿不到 `ball_query_dp_kernel_fast`、编译失败；宏 1 时现役 `bq_dp_kernel` 无条件保留、`ball_query_kernel_fast` 在 ballquery_kernel.cu 无条件定义，两实现同在一个翻译单元。
  原因：融合是**同一个插件 type（BallQueryGroup）的两个 launcher 实现**（宏切换后 type 名相同），无法像 tests_gridballquery.py 那样在同进程对比两个不同 type 名；也不宜在两个 .so 间对比（符号冲突）。
- **两路径必须写不同 buffer**：现役 idx/dp 与融合 idx/dp 各用独立内存，避免"自己比自己"恒通过。
- 用例矩阵：normalize_dp∈{0,1} × radius∈{0.1,10} × nsample∈{1,32,256} × N/M∈{N<nsample, N≈nsample, N≫nsample} × 对抗（同 voxel、重复坐标、精确球面 d²==r²、N=1、B=2）
- **空球用例单独处理（不参与 bit 级对拍）**：现役空球 idx 未初始化 → bq_dp 读垃圾 idx 算 dp；融合后 idx+dp 均不写。两者都是内存残留值，**不可能 max-abs-diff==0**。空球仅验证「用 compute-sanitizer 跑，无 out-of-bounds 报告」（CUDA 越界读不必然崩溃，"不崩溃"无法证明"无越界"），非空球用例才做 bit 级对拍。
- 判据（非空球用例）：dp max-abs-diff==0、idx bit 一致 100%。（gather 未改，idx 一致 ⟹ grouped 自动一致，故不对拍 grouped）

### V2 整网 E2E（参考 eval_gridballquery_miou.py）
- **engine 复用**：融合是 .so 内部实现变更，engine 文件（序列化图）不变；插件实现运行时从 .so 加载（trt_utils.py `ctypes.CDLL`）。故用**现役 engine 文件 + 融合 .so** 即可测融合路径，**无需重新 build engine**（省 ~92 分钟）
- fp32：ti10 acc == 0.9741 逐文件一致；全量 339 文件 acc == 0.9569
- 判据（fp32 与 fp16 同要求）：融合后与现役**逐点 pred 100% 一致（bit 级）**。融合是 bit 级等价（dp 计算顺序"先减后乘"与 bq_dp 一致、d² 平方符号无关），任何 <100% 都视为 bug（99.97% 是 fps_cache 档的近似度量，不适用于本融合）。fp16 引擎下 dp 恒 fp32、idx 恒 int32（supportsFormatCombination case 4），gather 未改，故 fp16 也要求 100% 逐点一致，**无 fp16 专属容差**
- 整网指标兜底（非主判据）：mIoU |Δ|≤0.5pp（仅防 engine 本身退化，不替代上述 100% 逐点判据）

### V3 性能（nsys A/B 对比，双编译同一 .cu）

- **双编译命令**：现役 .so 用现有 build 流程产出；融合 .so 用 `nvcc`/CMake 加 `-DHPENET_BQ_DP_FUSION=1`（或临时改 .cu 内 `#define HPENET_BQ_DP_FUSION 1` 重编）。两 .so 除该宏外完全一致，分别命名留存（如 `libhpenet_plugins.so` 与 `libhpenet_plugins_fusion.so`）
- **engine 复用**：同 V2，两个 .so 都加载同一现役 engine 文件（如 `deploy/hpenet_v2_fp32.engine`），只换 .so，无需重新 build engine
- 对两个 .so 各跑 `nsys profile`，`nsys stats --report gpukernsum`
- 采集三个量（单文件 6 子云口径，或单子云 N=2024/3523 口径）：
  - `T_bq` = 现役 `ball_query_kernel_fast` 总时间（基线 ~26.7%）
  - `T_dp` = 现役 `bq_dp_kernel` 总时间（基线 ~5.4%）
  - `T_fused` = 融合 `ball_query_dp_kernel_fast` 总时间
- 判据：净收益 = `(T_bq + T_dp) − T_fused`，须 > 0（融合整体更快）；预期 ~4–5%
- 若净收益 ≤ 0（融合后反而变慢），保持默认宏 0（现役），并记录结论

### V3.1 融合 1.5 触发条件（dp 写增量阈值）

定义 **dp 写增量** = `T_fused − T_bq`（融合 kernel 比纯搜索 kernel 多花的、主要由 dp strided 写导致的时间）。

**触发融合 1.5（dp 布局交错化）当且仅当**：

```
T_fused − T_bq  >  0.5 × T_dp
```

含义：融合 kernel 因 dp strided 写吃掉超过 **一半** 的 `bq_dp` 理论收益（即净收益缩水到 < 2.7% 量级），说明 strided 写已是主导瓶颈，值得付出布局变更的代价做交错化。

- 若 `T_fused − T_bq ≤ 0.5 × T_dp`：融合 1 已吃掉 dp 大部分开销，交错化收益有限，**不做**（维持 channel-first，避免下游 HPE/ONNX 改动）
- 若触发：立项融合 1.5，目标是把 dp 写增量压到 `≤ 0.3 × T_dp`（交错 float3 后重测）

以基线数字锚定：`T_dp ≈ 5.4%`、`T_bq ≈ 26.7%`，触发线即 **dp 写增量 > ~2.7%**。

## 收益与风险

- 收益：消除 1680 次 bq_dp launch（= 8 次/子云 × 210 子云）+ idx 全局读 + xyz 重读；预期 GPU kernel 时间 −4~5%
- 风险：融合 kernel 寄存器增加可能影响 occupancy；dp strided 写（3 段 stride=m*nsample）cache 不友好
- 缓解：V3 的 A/B 测试是最终裁决；纯实现等价优化，回退成本低

## TODOs

- [x] 1. 实现融合 kernel——在 `deploy/trt_plugins/src/ballquerygroup_kernel.cu` 顶部加 `HPENET_BQ_DP_FUSION` 宏（默认 0，`#ifndef` 保护）；新增 `ball_query_dp_kernel_fast`（定义包进 `#if HPENET_BQ_DP_FUSION`，严格复制 ballquery_kernel.cu:20-36 邻居搜索语义，只加 dp 写）；改 `ballquerygroup_launcher` / `ballquerydp_launcher` 用宏切换融合/现役两路径（现役路径原样调 `ball_query_launcher_with_stream` + `bq_dp_kernel`，**保留 bq_dp_kernel 和 `#include "ballquery_kernel.h"`**）
- [x] 2. V1 单元对拍——新建 `deploy/trt_plugins/tests/test_bq_dp_fusion.cu`（独立 nvcc 程序，`-DHPENET_BQ_DP_FUSION=1` 编译，include 两个 .cu，现役/融合两路径写不同 buffer），跑完用例矩阵（normalize_dp×radius×nsample×N/M×对抗），非空球 dp max-abs-diff==0 + idx bit 100% 一致；空球用 compute-sanitizer 验证无 out-of-bounds
- [x] 3. 双编译 .so——宏 0=现役 `libhpenet_plugins.so`、宏 1=融合 `libhpenet_plugins_fusion.so`（`-DHPENET_BQ_DP_FUSION=1`），两者除宏外完全一致
- [x] 4. V2 整网 E2E——现役 engine 文件 + 融合 .so（复用 engine 不重建），fp32/fp16 逐点 pred 100% 一致（bit 级）；ti10 acc==0.9741、全量 339 acc==0.9569 兜底
- [x] 5. V3 nsys A/B——两个 .so 加载同一 engine 各跑 nsys，采集 T_bq / T_dp / T_fused，净收益 = (T_bq+T_dp)−T_fused > 0，量化 dp 写增量
- [x] 6. 按 V3.1 阈值裁决

> **T6 裁决结论（基于 T5 实测，两条同时命中）**：
> - 净收益 = (T_bq+T_dp) − T_fused = −4,184,225 ns（**−0.49% < 0**）→ 融合 1 **不落地**，`HPENET_BQ_DP_FUSION` 保持默认 0（现役），fusion .so 留作对照档
> - dp 写增量 = T_fused − T_bq = 45,847,936 ns = **1.10 × T_dp** ≫ 阈值 0.5×T_dp → **立项融合 1.5（dp 布局交错化 float3）**，目标把增量压到 ≤0.3×T_dp
> - 实测数据：T_bq=208,970,493ns(24.57%)、T_dp=41,663,711ns(4.90%)、T_fused=254,818,429ns；engine=fps_algo_fps_cache_prune_fp32（Aug19 锚点）、真实文件 0000068.ply（6 子云×3523）、iters=30
> - 根因：strided dp 写（3 段 stride=m·nsample，channel-first 布局）是主导瓶颈，融合 kernel 比纯搜索慢 8%（grid=7）到 75%（grid=1），把 bq_dp 省的全部吃掉还倒贴——增量 ≤0.5×T_dp 且净收益>0 → 宏默认改 1 重走 V1/V2 落地；净收益≤0 → 保持默认 0 留对照档；增量 >0.5×T_dp → 立项融合 1.5（交错化）

## Final Verification Wave

- [x] F1. Plan compliance audit——逐任务核对 acceptance 全过、evidence 齐全（ledger 有 DoneClaim+AdversarialVerify），Must NOT 零违反（不改现役 ball_query_kernel_fast 签名、不删 bq_dp_kernel、不改 dp 布局、不碰 GridBallQuery/空球 memset）
- [x] F2. 代码质量终审——融合 kernel 与现役语义 bit 级等价复核（dp 符号/d² 判据/padding/early-break/normalize_dp）、无死代码、宏作用域正确
- [x] F3. 性能结论审查——V3 净收益数字真实（nsys 可复现）、V3.1 裁决执行到位（落地 or 回退 or 立项 1.5 有明确记录）

## 边界与不做什么

- 不改 ball_query_kernel_fast / ball_query_launcher_with_stream（旧 BallQuery 插件依赖）
- **不删除 bq_dp_kernel**（保留为现役回退/对照路径，宏切换）
- 不改 dp 内存布局（channel-first → 交错 float3 属"融合 1.5"，由 V3.1 阈值触发后才立项）
- 不碰 GridBallQuery 档（独立 kernel）
- 不碰空球 memset（§15.7 行为原样保留，融合后更安全但不主动改语义）
