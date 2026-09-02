# Orin 三口径延迟测量脚本（待落地）

> 创建：2026-08-27 | 状态：待执行（需 /start-work 落地到 deploy/，或用户手动复制）
> 背景：用户已把 `deploy/CPP_trt3` 交叉编译成 `hpenet_trt_infer`（fp32）与 `hpenet_trt_infer_fp16`，并在 Orin 上把 `hpenet_v2_plugin.onnx` 转成 fp32/fp16 engine。CPP 默认 voxel_size=0.02、data_dir=radarfullwl/raw 已改好。需测 100 帧的三口径延迟：端到端 / 部署（去 PLY_LOAD）/ GPU kernel 构成。
> 约束：零 git 操作；脚本在 Orin（aarch64，JetPack）上运行，nsys 为 JetPack 自带。
> 审查：Oracle 两轮已审（对照真实 cpp_v002.sqlite schema + 本机 nsys 实测）。**schema/SQL/nsys 旗标全部正确**。已修正：① cutoff 用「首个 PLY_LOAD」而非「第 6 个 FPS」（后者漏 file1 的 PLY_LOAD/VOXELIZE，端到端/部署系统性偏低 ~9%；修正后 16.216/6.689/4.798ms 精确对上二进制计时）；② nsys stats 加 `--force-overwrite=true`；③ stats 失败 `|| echo` 防 set -e 静默终止；④ **profile 前 `rm -f *.sqlite`**（二轮复审发现：同名重跑会读到陈旧 sqlite，stats 沿用不重导出 → 静默错误）；⑤ analyze 用 PLY_LOAD 计数自推实际文件数。另补 None/除零/db 存在/NUM_FILES 防御。

## 待办

- [x] 1. 落成 `deploy/measure_orin.sh`（内容见下）
- [x] 2. 落成 `deploy/analyze_latency.py`（内容见下）
- [x] 3. `chmod +x deploy/measure_orin.sh`（可选，或 bash 直接跑）

## 脚本 1: deploy/measure_orin.sh

```bash
#!/bin/bash
# ============================================================
# HPENet V2 — Orin 三口径延迟测量（一次 nsys 采集全口径）
# 口径1 端到端 benchmark（含 PLY IO）
# 口径2 部署口径（去 PLY_LOAD，见 analyze_latency.py）
# 口径3 GPU kernel 构成（gpukernsum）
#
# 用法（Orin 上）:
#   bash deploy/measure_orin.sh
#   python3 deploy/analyze_latency.py /tmp/opencode/orin_lat/e2e_fp32.nsys-rep [NUM_FILES]
# ============================================================
set -euo pipefail

# ── 配置（按实际路径改）──
BIN_FP32=./hpenet_trt_infer
BIN_FP16=./hpenet_trt_infer_fp16
ENGINE_FP32=hpenet_v2_fp32.engine
ENGINE_FP16=hpenet_v2_fp16.engine
# voxel_size / data_dir 已在 CPP main.cpp 默认改好（0.02 / radarfullwl/raw），
# 如需覆盖再显式加 --voxel_size=0.02 --data_dir=...
NUM_FILES=100
OUT=/tmp/opencode/orin_lat
mkdir -p "$OUT"

# 若插件/TRT/cudnn 非默认路径，取消注释并补全：
# export LD_LIBRARY_PATH=/path/to/cudnn/lib:/path/to/libnvinfer.so:$LD_LIBRARY_PATH

command -v nsys >/dev/null || { echo "错误: 找不到 nsys（JetPack 自带，检查 PATH）"; exit 1; }

measure() {  # $1=名称  $2=二进制  $3=engine
  local name=$1 bin=$2 eng=$3
  echo "=== [$name] 测量中 (${NUM_FILES} 帧) ==="
  # 先删旧 sqlite：nsys stats 遇同名 .sqlite 会直接沿用（不重导出），
  # 同名重跑会读到上一次 profile 的 sqlite → 结果陈旧。删掉后 stats 从 .nsys-rep 重新导出。
  rm -f "$OUT/$name.sqlite"
  nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
    -o "$OUT/$name" \
    "$bin" --engine="$eng" --num_files="$NUM_FILES" \
    | tee "$OUT/$name.log"
  for rpt in gpukernsum cudaapisum nvtxkernsum; do
    nsys stats --report "$rpt" --format csv --force-overwrite=true \
      --output "$OUT/$name" "$OUT/$name.nsys-rep" 2>/dev/null \
      || echo "⚠️ [$name] $rpt stats 失败（继续）"
  done
  echo "=== [$name] 完成 ==="
}

measure e2e_fp32 "$BIN_FP32" "$ENGINE_FP32"
measure e2e_fp16 "$BIN_FP16" "$ENGINE_FP16"

echo "产出在 $OUT/"
ls -la "$OUT"
```

## 脚本 2: deploy/analyze_latency.py

```python
#!/usr/bin/env python3
# ============================================================
# HPENet V2 — nsys 延迟分析（三口径）
#   - GPU kernel 总时间 + 构成（口径3）
#   - NVTX 分段 → 端到端（口径1）/ 部署口径（口径2 = 去 PLY_LOAD）
#
# 用法:
#   python3 analyze_latency.py /tmp/opencode/orin_lat/e2e_fp32.nsys-rep [NUM_FILES]
#   （NUM_FILES 默认 100；nsys 生成同名 .sqlite 需在同目录）
# ============================================================
import sqlite3
import os
import sys

rep = sys.argv[1]
NUM_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
if NUM_FILES <= 0:
    sys.exit(f"错误: NUM_FILES 必须 > 0，得到 {NUM_FILES}")
db = rep.replace('.nsys-rep', '.sqlite')
if not os.path.exists(db):
    sys.exit(f"错误: 找不到 sqlite 文件 {db}（nsys profile 会生成同名 .sqlite）")
con = sqlite3.connect(db)
cur = con.cursor()

# 推理起点 = 首个 PLY_LOAD 的 start。
# warmup 用随机输入、不读文件（无 PLY_LOAD），故首个 PLY_LOAD 天然是第一个真实文件的起点，
# 且位于 file1 的 voxelize 内核之前 —— 比「第 6 个 FPS 实例」更完整
# （后者落在 file1 SUBCLOUD_LOOP 内部，会漏 file1 的 PLY_LOAD/VOXELIZE，端到端偏低 ~9%）。
cur.execute(
    "SELECT start FROM NVTX_EVENTS WHERE text = 'PLY_LOAD' ORDER BY start LIMIT 1"
)
row = cur.fetchone()
if not row:
    sys.exit("错误: 未找到 PLY_LOAD NVTX 事件（profile 为空或未 trace nvtx）")
infer_start = row[0]
print(f'推理起点（首个 PLY_LOAD）: {infer_start}')

# 实际文件数自推：每个真实文件恰好一个 PLY_LOAD 事件（warmup 无 PLY_LOAD）
cur.execute(
    "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text = 'PLY_LOAD' AND start >= ?",
    (infer_start,),
)
actual_files = cur.fetchone()[0]
if actual_files and actual_files != NUM_FILES:
    print(f'⚠️ 传入 NUM_FILES={NUM_FILES}，但实际 PLY_LOAD 计数={actual_files}，改用实际值')
    NUM_FILES = actual_files

# GPU kernel 总时间（推理区间）
cur.execute(
    "SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ?",
    (infer_start,),
)
gpu_ms = (cur.fetchone()[0] or 0) / 1e6
print(f'GPU kernel 总（推理区间）: {gpu_ms:.3f} ms → 每文件 {gpu_ms / NUM_FILES:.3f} ms')

# GPU kernel 构成（口径3）
cur.execute(
    """
    SELECT s.value, SUM(k.end - k.start) / 1e6, COUNT(*)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON s.id = k.demangledName
    WHERE k.start >= ?
    GROUP BY k.demangledName
    ORDER BY 2 DESC LIMIT 8
    """,
    (infer_start,),
)
print('\nGPU kernel 构成（推理区间，按时间降序 Top8）:')
for name, ms, n in cur.fetchall():
    pct = (ms / gpu_ms * 100) if gpu_ms > 0 else 0.0
    print(f'  {ms:7.3f} ms  {pct:4.1f}%  n={n}  {name.split("::")[-1][:55]}')

# NVTX 分段（口径1 端到端 / 口径2 部署）
cur.execute(
    """
    SELECT text, SUM(end - start) / 1e6
    FROM NVTX_EVENTS
    WHERE start >= ? AND text IN
      ('PLY_LOAD', 'VOXELIZE', 'SUBCLOUD_LOOP', 'TAIL', 'COORD_SHIFT', 'ARGMAX_ACC')
    GROUP BY text ORDER BY 2 DESC
    """,
    (infer_start,),
)
seg = {name: (ms or 0) for name, ms in cur.fetchall()}
tot = sum(seg.values())

print('\nNVTX 分段（每文件）:')
for name in ['PLY_LOAD', 'SUBCLOUD_LOOP', 'TAIL', 'VOXELIZE', 'COORD_SHIFT', 'ARGMAX_ACC']:
    if name in seg:
        print(f'  {name:16s} {seg[name] / NUM_FILES:6.3f} ms')

e2e = tot / NUM_FILES
deploy = (tot - seg.get('PLY_LOAD', 0)) / NUM_FILES
print(f'\n端到端（分段和）:      {e2e:.3f} ms/文件')
print(f'部署口径（去 PLY_LOAD）: {deploy:.3f} ms/帧')
```

## 落地说明

- 脚本完整内容如上，用户可**直接复制**到 Orin 上使用（`bash measure_orin.sh` + `python3 analyze_latency.py ...`），无需等落地。
- 若要正式落成 `deploy/measure_orin.sh` + `deploy/analyze_latency.py` 文件，需 `/start-work` 执行本计划。
- **关键修正（vs 初版）**：cutoff 用「首个 PLY_LOAD」而非「第 6 个 FPS」，否则端到端/部署口径系统性偏低 ~9%（漏 file1 的 PLY_LOAD/VOXELIZE）；nsys stats 加 `--force-overwrite=true` 防重跑陈旧 CSV。
- 注意：`analyze_latency.py` 的 per-file 均值用 `NUM_FILES` 参数（默认 100）；若数据目录可用文件 < 100，`process_directory` 会按 `min(num_files, n_test)` 截断，此时应传实际文件数（或用 NVTX PLY_LOAD 计数自推）。
