# ball_query 距离 fp16 半精度化 — 验证方案

> 创建：2026-08-31 | 状态：**执行完成（阶段 1 + 5b）——结论：fp16 半精度化在当前坐标表示下不可行**
> **实测结果（2026-08-31）**：① 5b pos 范围 = **max|pos|≈200m**（195.879/211.276m）→ 判据「不可接受」（fp16 粒度 9.8cm ≫ voxel 2cm）；② 阶段 1 fp16 距离计算 acc 损失 **−0.03pp**（0.9577→0.9574）、真实口径 idx 翻转率 **0%** → 闸门「通过」。**综合：阶段 1 精度够但无性能（坐标仍 fp32 读入）、阶段 2 有性能但精度崩（pos 200m）——fp16 两不讨好，须先做坐标缩放归一化（offset+scale）才可行。**
> 背景：Orin 实测（2026-08-31）ball_query 是**第一大 GPU kernel（27%，4.78ms）**，带宽退化 3.5×（L20 1.38ms→Orin 4.78ms）。半精度化是潜在优化方向，但 fp16 尾数仅 10 位，距离计算的 ~0.1% 舍入会落在 `d2 < radius2` 的**边界比较**上，可能翻转邻居选择。本方案**先量化精度损失，再量化性能提升**，分两阶段，各自独立验收。
> 约束：零 git 操作；改动仅限 `deploy/trt_plugins/src/ballquery_kernel.cu`（TRT 插件主路径）；不改 `openpoints/cpp/pointnet2_batch/src/ball_query_gpu.cu`（Python 训练侧，勿动）。

---

## 一、现状锚点（源码已确认）

**调用链澄清（重要）**：ONNX 用的是 `BallQueryGroup`（BQG）/ `BallQueryDP`（BQD）op，但它们的搜索内核**最终都委托到 `ballquery_kernel.cu` 的 `ball_query_kernel_fast`**：

```
BQG/BQD 插件 enqueue → ballquerygroup_launcher / ballquerydp_launcher（ballquerygroup_kernel.cu L119/L159）
  → 宏 HPENET_BQ_DP_FUSION=0（现役）→ ball_query_launcher_with_stream（ballquery_kernel.cu L39）
  → ball_query_kernel_fast（ballquery_kernel.cu L5）← 距离计算唯一在此
```

- `ball_query_kernel_fast` 唯一一份定义在 `ballquery_kernel.cu` L5；`ballquerygroup_kernel.cu` 经 `#include "ballquery_kernel.h"`（L9）复用。
- **例外（不改）**：① 融合 kernel `ball_query_dp_kernel_fast`（定义在 `ballquerygroup_kernel.cu` **L24**，发射点在 L130/L167；宏 `HPENET_BQ_DP_FUSION=1`）是独立搜索+dp，**其 L43 有第二份距离计算**，已否决（净收益 −0.49%），现役宏 0 不用；② `gridballquery_kernel.cu` L225 的 grid_query 是 GridBallQuery（已否决）第三份距离计算。两处均不激活、不改。

`deploy/trt_plugins/src/ballquery_kernel.cu` 的 `ball_query_kernel_fast`：

```c
// L15-37 关键部分
float radius2 = radius * radius;
float new_x = new_xyz[0], new_y = new_xyz[1], new_z = new_xyz[2];
...
for (int k = 0; k < n; ++k) {
    float x = xyz[k*3+0], y = xyz[k*3+1], z = xyz[k*3+2];
    float d2 = (new_x-x)*(new_x-x) + (new_y-y)*(new_y-y) + (new_z-z)*(new_z-z);  // L25 ← 距离计算
    if (d2 < radius2) { ... idx[cnt] = k; ++cnt; if (cnt>=nsample) break; }
}
```

- 输入 `new_xyz`/`xyz` 为 **fp32** 坐标；`radius`/`radius2` 为 fp32。
- 距离计算 L25 是纯 fp32。
- 关键语义：`d2 < radius2`（严格小于）+ `cnt==0` 时首邻居填充 + `cnt>=nsample` 早停（升序扫、取索引最小的 nsample 个）。

---

## 二、阶段 1：精度验证（最小改动，只改距离计算）

### 2.1 改动（ballquery_kernel.cu L25，宏切换，默认关）

在 `ballquery_kernel.cu` 顶部加 `#ifndef BALLQUERY_FP16 / #define BALLQUERY_FP16 0 / #endif`，L25 处用宏切两条路径：

```c
#if BALLQUERY_FP16
    // fp16 距离：坐标差转 __half（⚠️ 舍入的是「差值」，≠ 阶段 2 的「坐标」fp16 存储；
    // 两者精度模型不同，阶段 1 的 acc 是阶段 2 的「上界」而非预测值，见 §三），
    // 平方和用 __hfma（融合乘加，单次舍入，共 3 次 fp16 舍入，比逐项舍入更精确）
    __half dx = __float2half_rn(new_x - x);
    __half dy = __float2half_rn(new_y - y);
    __half dz = __float2half_rn(new_z - z);
    float d2 = __half2float(__hfma(dx, dx, __hfma(dy, dy, __hmul(dz, dz))));
#else
    float d2 = (new_x-x)*(new_x-x) + (new_y-y)*(new_y-y) + (new_z-z)*(new_z-z);
#endif
```

- 默认 `BALLQUERY_FP16=0`（现役 fp32 行为零变化，回退通道）。
- 阶段 1 只回答「精度损失多大」，**不追求性能**（坐标仍 fp32 读入，性能提升留到阶段 2）。

### 2.2 验证步骤

1. **编译**：宏 0 与宏 1 各编译一次 `libhpenet_plugins.so`（或独立 .so 对照，参考 FPS warp 的双编译流程）。
2. **kernel 级对拍（量化 idx 翻转率）**：新建独立 nvcc 对拍程序（照 `test_bq_dp_fusion.cu` 的 `#include` 模式；因 `ballquery_kernel.cu` 是 include 进测试程序、宏在 include 期固定，单一二进制无法同时承载两路径——**需两份独立编译的二进制各 dump idx，再 host diff**）。统计**两个口径**的翻转率：①「受影响 query 点数 / 总 query 点数」（点级）；②「不一致 slot 数 / 总 slot 数」（slot 级——因早停+首邻居填充会级联移位，slot 级会高估真实翻转）。
3. **整网 acc（量化精度损失）**：
   - L20：`python deploy/trt_inference.py --num_files 10 --voxel_size=0.02`（或 CPP 二进制），fp32 基线 acc 0.9578（ti10）对比 fp16 版；
   - Orin：`./hpenet_trt_infer --num_files 100 --voxel_size=0.02`，基线 0.9425（100 文件）对比 fp16 版。
4. **记录**：翻转率 + 两平台 acc 差值，写入 `.omo/notepads/`。

### 2.3 验收标准（阶段 1 闸门）

- **翻转率**：记录实测值（预期 <1%）。
- **acc 损失**：**<0.3pp 为可接受**（预期 <0.2pp）；≥0.3pp → 半精度化**否决**，宏保持 0，负结果归档。
- **bit 级一致性不要求**（fp16 舍入 ≠ fp32，bit 级必然破坏，判据降级为 acc 逐字符 + 翻转率）。

---

## 三、阶段 2：性能验证（fp16 存储 + half2 打包）

> **仅在阶段 1 通过（acc 损失 <0.3pp）后执行**。

### 3.1 改动（真正半精度化，追求性能）

- 输入坐标 `new_xyz`/`xyz` 改为 **fp16 存储**（上游 pos 生成处转 `__half`，TRT 插件 `configurePlugin` 输入类型改 kHALF）；
- 距离计算用 **half2 打包**（两个 half 一次 SIMD 算），内存带宽减半 + half2 吞吐翻倍；
- 这是较大改动（涉及 `ballquery_plugin.cpp` 的 `configurePlugin`/`supportsFormatCombination`、上游 pos 类型、reformat 层）。

### 3.2 验证步骤

- **前置（pos 范围调查，O1）**：`pos` 已是中心化坐标（`preprocess_subcloud` step 2 减 min + step 3 减 mean 中心化，见 preprocessor.cpp），故调查对象是 **pos 的 max|·|（= 子云半径）**，不是原始 PLY 坐标。
  **测量方式**：在 `preprocessor.cpp` 的 `result.pos` 填充后（step 7 附近）加**临时**调试打印 `max|result.pos|`，跑一次推理（L20 或 Orin，`--num_files 2`），收集各子云半径分布，测完删打印。
  **判据**（fp16 相对精度 2⁻¹¹，绝对粒度 ≈ R×2⁻¹¹；voxel 0.02）：
  - max|pos| **< 20m** → 粒度 < 1cm < voxel/2，fp16 坐标存储**可接受**；
  - **20~50m** → 粒度 1~2.5cm，**边界**（需实测 acc 确认）；
  - **> 50m** → 粒度 > 2.5cm（超 1 个 voxel），**不可接受**，阶段 2 需另做更细中心化或放弃 fp16 坐标存储。
- **溢出分析（O2，良性但须记录）**：fp16 max≈65504，`dx>~256m` 时 `dx²` 溢出为 inf。但溢出点的真实 `d2>>radius2`（radius=5→25），`inf<25` 为 false 正确剔除，与 fp32 一致，不产生假阳性/假阴性——前提是 radius 远小于 256 且坐标幅度有限，须归档此前提。
- nsys 对比 `ball_query_kernel_fast` 的 GPU 时间（fp32 vs fp16 存储），目标 **>1.5× 提速**；
- 整网 acc 复验（fp16 坐标存储的 acc **可能比阶段 1 更差**——阶段 1 舍入「差值」、阶段 2 舍入「坐标」，灾难性抵消使后者更差；**阶段 1 的 acc 是上界，非阶段 2 的预测值**）。

### 3.3 验收标准（阶段 2）

- ball_query kernel 提速 **>1.5×** 且整网部署口径下降可测；
- acc 损失仍 <0.3pp。

---

## 四、边界与止损

- **不改**：`openpoints/cpp/pointnet2_batch/src/ball_query_gpu.cu`（训练侧）、`ballquerygroup_kernel.cu`（GridBallQuery/融合 1 对照路径，已否决，勿动）、`fps_kernel.cu`（本方案不涉 FPS）、`scatter_mean`（原子加不可 fp16）。
- **零 git 操作**；宏默认 0，回退 = 宏改回 0 重编。
- **阶段 1 失败**（acc ≥0.3pp）→ 半精度化否决，负结果归档，不进入阶段 2。
- **阶段 2 失败**（提速 <1.5× 或 acc 超阈）→ 回退 fp32 存储，只保留阶段 1 的结论（fp16 距离计算精度损失数据）。

## 五、待办

- [x] 1. ballquery_kernel.cu 加 `BALLQUERY_FP16` 宏（默认 0）+ L25 距离计算双路径（含 __half2float/__hfma 语义）
- [x] 2. 双编译 .so（宏 0 现役 / 宏 1 fp16），md5/SASS 验证区分（nm 符号名一致，靠 cuobjdump SASS 半精度指令 + md5 区分）
- [x] 3. kernel 级对拍：idx 翻转率量化（真实口径 0.00%，密集加压 1~2% 级联实锤）
- [x] 4. 整网 acc：L20 ti10，fp32 0.9577 vs fp16 0.9574（−0.03pp）
- [x] 5. 阶段 1 闸门判定：0.03pp < 0.3pp → 通过
- [x] 5b. **pos 范围调查（O1 前置）**：max|pos| = 195.879/211.276m → **不可接受（>50m）**，fp16 粒度 9.8cm 太粗
- [ ] 6.（阶段 2）~~fp16 存储 + half2 打包 + TRT 插件接口改 kHALF~~ **不可行，不执行**（pos 200m 致 fp16 坐标存储精度崩）
- [ ] 7.（阶段 2）~~nsys ball_query 提速验证~~ **不执行**
