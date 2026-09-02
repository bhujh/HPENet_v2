# ball_query 距离 fp16 半精度化 — 阶段 1 精度验证结论归档

> 归档日期：2026-09-01 | 状态：**阶段 1 通过（acc 损失 0.03pp < 0.3pp 闸门）**，可进入阶段 2

## 结论

`ball_query_kernel_fast` 距离计算 fp16 化（`BALLQUERY_FP16` 宏）在真实部署口径下**精度损失 0.03pp**（0.9577 → 0.9574），远低于 0.3pp 闸门。**闸门通过，阶段 2（fp16 存储 + half2 打包）可继续。**

## 改动（ballquery_kernel.cu）

- 顶部加 `#ifndef BALLQUERY_FP16 / #define BALLQUERY_FP16 0 / #endif` + `#include <cuda_fp16.h>`。
- L32-39 距离计算双路径：宏 1 走 `__float2half_rn(差值)` + `__hfma/__hmul` 平方和；宏 0 走原 fp32。默认 0（回退通道）。
- SASS 确认 fp16 路径被编译器自动向量化为 `F2FP.F16.F32.PACK_AB` + `HMUL2` + `HFMA2`（half2 打包，语义正确）。

## 双 .so（nm/size/md5 区分）

| .so | 宏 | md5 | 距离计算 SASS |
|---|---|---|---|
| `libhpenet_plugins.so`（现役） | 0 | `86056ad7…` | `FMUL`/`FFMA`（float） |
| `libhpenet_plugins_fp16.so` | 1 | `c6169db5…` | `F2FP`/`HMUL2`/`HFMA2`（half） |

- `nm -C` 两者符号名一致（`ball_query_kernel_fast` 签名未变，nm 无法按符号名区分）。
- 实际区分靠 **cuobjdump SASS**（half vs float 指令）+ md5。尺寸同为 1349904 B（代码量巧合相等）。
- 最终 `libhpenet_plugins.so` 已恢复宏 0 现役版（md5 `86056ad7…` 验证）。

## kernel 级 idx 翻转率（点级 / slot 级）

| scenario | 点级% | slot级% | 说明 |
|---|---|---|---|
| voxel_dense_r5 | 2.34 | 1.02 | 密集加压（非真实） |
| uniform_dense_r5 | 1.17 | 0.40 | 密集加压 |
| uniform_large_r5 | 2.00 | 0.99 | 密集加压 |
| uniform_small_r01 | 0.10 | 0.006 | 小 radius |
| uniform_r5_S8 | 0.00 | 0.00 | S=8 |
| boundary_shell_r5 | 0.78 | 0.32 | 对抗性边界壳（上界） |
| **uniform_sparse_r5** | **0.00** | **0.00** | **真实部署口径（~5000 点/300m 盒）** |

- **真实部署口径翻转率 = 0%**（0/4000 query，0/128000 slot）。
- 密集场景的 1~2% 是人为加压（~33 邻居/球），真实子云稀疏（~0.1 邻居/球，radius 5 内大多空球/1 邻居），边界翻转几乎不发生。
- 级联移位实锤：voxel_dense 24 受影响 query → 333 slot 差异（~14×），slot 级高估。

## 整网 acc（L20，voxel 0.02，ti10，`hpenet_v2_fp32.engine`）

| 版本 | mean acc |
|---|---|
| fp32 基线（宏 0） | **0.9577**（复现文档基线 0.9578） |
| fp16 版（宏 1） | **0.9574** |
| **差异** | **−0.03pp** |

- 逐文件：7 降 2 升 1 平，最大 |Δ| = 0.0009（0000076），纯噪声级。

## 闸门判定

**acc 损失 0.03pp < 0.3pp → 阶段 1 通过**。fp16 距离计算精度损失可忽略，进入阶段 2（fp16 坐标存储 + half2 打包 + TRT 接口 kHALF），但需先做 O1 pos 范围调查（§三）。

## 复现命令

```bash
# 双 .so（宏改值 → make → cp → 宏改回 → make）
cd deploy/trt_plugins/build && make -j

# kernel 对拍
cd deploy/trt_plugins/tests
nvcc -O2 -arch=sm_89 -I../include test_bq_fp16.cu -o test_bq_fp16_fp32
nvcc -O2 -arch=sm_89 -I../include -DBALLQUERY_FP16=1 test_bq_fp16.cu -o test_bq_fp16_fp16
./test_bq_fp16_fp32 idx_fp32.bin && ./test_bq_fp16_fp16 idx_fp16.bin
python3 compare_bq_fp16.py idx_fp32.bin idx_fp16.bin

# 整网 acc（swap 插件 .so 后跑两次）
python deploy/trt_inference.py --num_files 10 --voxel_size=0.02
```
