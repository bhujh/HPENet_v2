# learnings — omitting-voxelize

## P2 C++ 基线留档（2026-09-16，GPU 5 / L20，无 git 操作，源文件零改动）

### 构建（clean rebuild）
- 文档口径命令 `cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89" -DCMAKE_BUILD_TYPE=Release` → exit 0；但 `make -j` → **exit 2**（链接失败：libcudnn.so.8 找不到，libnvinfer_plugin.so 的传递 NEEDED）。
- 补救（唯一偏差，不改源）：追加 `-DCUDNN_LIB=/home/wangpeng/miniforge3/envs/hpenet/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8` → cmake exit 0、make -j exit 0。
- 原因：本机无系统 cuDNN；pip nvidia-cudnn-cu11 只有 `libcudnn.so.8`（无 `libcudnn.so`）→ CMakeLists.txt:76-85 两条探测路径均落空。**C7 clean rebuild 必须带 -DCUDNN_LIB 或等效环境准备。**
- 工具链：cmake 3.28.3 / gcc 11.5.0 / nvcc 11.8.89 / TRT 8.6.1.6；编译警告仅 `cuda_utils.h:26 unused template param T`×5（无害）。
- 产物：`build/hpenet_trt_infer` 3,777,896 B；sha256 `0b52d744209a3e05efb2f40b0b5fc4731aae1c36f529eb803bf269d1d535fbc7`；md5 `11c286101d340d213fd01d68fd89951f`。plugins 静态链接 `build/hpenet_plugins_build/libhpenet_plugins.a`（1,429,650 B；clean build 不再产出旧 build 根目录的 libhpenet_plugins.so）。
- **运行期硬约束**：必须 `LD_LIBRARY_PATH` 含 pip cudnn 目录（libnvinfer_plugin.so.8 的传递 NEEDED 不经 RUNPATH 解析）——否则连 `--help` 都 exit 127。
- 新 link.txt 仍引用 `src/voxelizer.cu.o`、`src/fnv_hash.cu.o` → **C7 删除文件后只 `make` 必失败，必须 clean rebuild**（证实 spec §4 方案 C item 4）。

### CLI 基线（--num_files 10，默认 min_n=2024/max_n=10000/voxel_size=0.0001f/warmup=5/seed=100）
- 文件选择：sorted → test_start=int(339*0.2)=67 → 本次为 `0000068.ply`..`0000077.ply`。
- run1（无 profiler）：逐文件 acc 0.9396/0.9294/0.9456/0.9204/0.9460/0.9378/0.9449/0.9408/0.9142/0.9388，**Mean accuracy 0.9358**；逐文件 Time(s, 3位小数) 0.010/0.009/0.009/0.009/0.008/0.009/0.009/0.009/0.010/0.010，**Mean time 0.009s/文件**。CLI **打印逐文件 acc**（process_file 的 label 对比）。
- run2（nsys 下，acc 与 run1 逐文件完全一致 → 推理确定性）：Mean time 0.011s。
- nsys NVTX（analyze_latency.py, 10 文件）：端到端 10.455 ms/文件；**部署口径（去 PLY_LOAD）4.247 ms/帧**；**VOXELIZE 0.462 ms/文件**（= C 删除段，F4 基线）；PLY_LOAD 6.208 / SUBCLOUD_LOOP 2.460 / TAIL 1.312 / COORD_SHIFT 0.007 / ARGMAX_ACC 0.007；GPU kernel 总 2.686 ms/文件（FPS warp 39.8% + ball_query 25.5%）。profile 存 `/tmp/opencode/cpp_trt4_baseline.{nsys-rep,sqlite}`。
- `subcloud=1` 语义：voxel 0.0001 → 每点独立体素 → max_count=1 → `idx_points` 恒 1 个子云（全体点 shuffle 一遍）→ 去体素化后单子云「由构造保证」，与 spec 一致。
- GPU 显存 386 MiB used / 45459 MiB total；TRT WARN：链接 cuDNN 8.9 vs 加载 8.7（pip 版本），不影响推理。

### 磁盘构件（实测，勿信 AGENTS.md 旧值）
- engine：`deploy/hpenet_v2_fp32.engine` **14,679,492 B**，2026-09-07 20:42（AGENTS.md 写 14,446,844 B / 08-26 已过时；时间与钉死 ckpt run 20260907-170521 吻合，推测由其导出，本次未验证 pth 出处）。
- onnx：`deploy/hpenet_v2_plugin.onnx` 11,899,013 B，2026-09-07 20:41。
- stats：`deploy/CPP_trt/stats_feat5.json` 296 B（feat_mean/std 4 维 + z_mean/z_std）。
- 数据：`data/RadarClassi/radarfullwl/raw` 339 个 PLY，99 MB。

### CLI 参数面（C7 复测用）
`--engine` / `--stats` / `--data_dir` / `--num_files` / `--min_n` / `--max_n` / `--voxel_size` / `--warmup` / `--output` / `--benchmark` / `--seed` / `--dump-prefix` / `--help`，支持 `--key=value` 与 `--key value` 两种形式。注意：`--output`、`--benchmark` 解析后 CLI 路径不使用（output 仅 C-API 写 PLY）；help 文案称 voxel_size 默认 0.02，代码实际默认 0.0001f（main.cpp:31）。C7 删 voxel_size 后调用方式：去掉该参数即可，其余不变（同一 data_dir → 同 10 文件）。


## P1 Python 基线重测（2026-09-16，GPU 3 / L20，无 git 操作，源文件零改动）

### 测量命令（A4/F2 必须逐字符复用）
```
source /home/wangpeng/miniforge3/etc/profile.d/conda.sh && conda activate hpenet && CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml wandb.use_wandb=False mode=test --pretrained_path log/radar/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV/checkpoint/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV_ckpt_best.pth
```

### 结果（两次复跑，确定性验证）
- run1（45s）/ run2（35s）：`test_oa 92.05 / test_macc 88.50 / test_miou 77.02 / mp 84.55 / mr 88.50`，iou [90.79, 63.26]，prec [96.60, 72.51]，rec [93.79, 83.21]，Best ckpt @E85。两次 CSV 逐位一致，仅 **iou_invalid 2 位小数边界抖动 63.26/63.25**（原始值第 3 位附近）→ **非 bit 级确定，但 2dp 口径确定性成立**（OA/mACC/mIoU 两次完全一致）。来源疑为 cuDNN/CUDA 归约非确定（cfg.deterministic=false）。
- 钉死 ckpt：35,805,456 B，sha256 `130340c92bdd4dae0ff5a3331be9656eb69b7a7109181bbc20b697182e42f1ec`；加载无错，best_epoch=85。
- 生效 seed：`main.py:606 set_random_seed(0)` 在 test() 内重置（data loop 前）→ 每文件 shuffle/FPS 起点可复现。

### ti10 定义（本任务确认）
- **ti10 只定义在部署/TRT 路径**（`deploy/trt_inference.py:232-238` 与 C++ `list_test_files`）：sorted(raw 339) → `int(n*0.2)=67` 起 → 前 10 = **0000068..0000077.ply**；metric = 逐文件 OA 的 10 文件 mean（`(pred==label).mean()`）。
- Python `mode=test` 无 ti10 概念：`generate_data_list` 用 seed(100) shuffle + 后 17% → 58 文件全量；故**Python 参考以 58 文件全量口径为准**（上表 92.05/88.50/77.02）。
- 补充测量（/tmp/opencode/ti10_baseline.py，复用 main.py load_data/test 代码路径，不改仓库）：Python 参考在 ti10 文件集上的 **10 文件 mean OA = 93.2147 / 93.2171**（两次），pooled OA 93.19 / mIoU 80.10 / mACC 90.49；9/10 文件逐位一致，仅 0000069 抖动 ~0.02pp。⚠️ 勿与 C++ CLI 基线混淆：CLI 用 FPSPrune 引擎（P2 节 mean 0.9358）与 Python 无 prune 模型是不同变体。

### 副作用与口径注意（勿再犯）
- `mode=test` 的 `resume_exp_directory` **不会新建顶层 run 目录**，而是把产物写回 pretrained 所在目录：新增 `*_test.csv` + 带时间戳 `.log`（`...N3jVvf3C6aJCqvS9rLQ6vV20260916-134552-*.log` / `...-134705-*.log`），并**覆盖该目录 cfg.yaml / 重拷 hpenet-ll.yaml**（框架行为，非本任务修改；voxel_size 0.0001 在覆盖后仍保留，ckpt/训练 CSV/训练 log 未动）。A4 复测时预期同样落回该目录。
- 58 文件全量 test 集的逐文件指标（cloud 0..57）在 run log 中可查，含 per-file OA/mIoU/mACC；如需逐文件对照（F2 口径），直接 grep 两次 log 即可，无需重跑。

## A2 落地（2026-09-16，checkbox 4）
- `cfgs/radar/default.yaml:7`：`voxel_size: 0.0001 #...` → `voxel_size: null #...`（历史注释逐字保留）。仅改此标量，未动 line 10/15/19 的 `voxel_max`。
- 解析验证：`python3 -c "import yaml;d=yaml.safe_load(open('cfgs/radar/default.yaml'))['dataset']['common'];print(repr(d['voxel_size']))"` → `None`。
- `cfgs/radar/hpenet-ll.yaml` grep `voxel_size`：**无匹配** → 该文件不覆盖 voxel_size，A2 生效路径不被破坏。
- 旁证：本 notepad 第 46 行「ti10 定义」指出 `mode=test` 无 ti10；A2 后的全量复测口径沿用 P1 节 58 文件（OA 92.05 / mACC 88.50 / mIoU 77.02）。
- 未跑训练/测试（A1 f-string 修复未落地前 `voxel_size: null` 会触发 `TypeError: unsupported format string passed to NoneType.__format__`，属预期，非本任务缺陷）。

## A1 完成：s3disRadar.py 的 voxel_size 标签条件化（2026-09-16）

- 文件：`openpoints/dataset/radar/s3disRadar.py`（仅此一处改动，`s3disRadar_sphere.py` 未动）。
- 改动：原 line 73-74 的 f-string 直接 `{voxel_size:.3f}` → 新增 line 73 `tag = 'novx' if voxel_size is None else f'{voxel_size:.3f}'`，f-string 改用 `{tag}`（`filename` 在 74-75 行，其余不变）。
- 崩溃点消除：`voxel_size=None`（A2 将把 `cfgs/radar/default.yaml:7` 设为 null）时旧代码 `f'{None:.3f}'` 抛 TypeError；新代码 tag=`novx`。该 filename 在 `if presample` 之前无条件构建 → train/val/test 三模式都命中，必须修，已修。
- 浮点行为零变化（实测）：v=0.0001 → `0.000`；v=0.02 → `0.020`（`f'{v:.3f}'` 语义保持）。
- 附带收益：旧 tag 把 0.0001..0.0009 都渲染为 `"0.000"`（缓存碰撞隐患），novx 为字面量，不复用 `str(voxel_size)`。
- 验证：`python3 -m py_compile openpoints/dataset/radar/s3disRadar.py` → exit 0；两个分支以真实解释器证明（novx / 0.000）。
- 未运行任何 git 命令；未跑训练/测试管线。

## A3③ 完成：onnx_inference.py:267 BLK-2 硬崩修复（2026-09-16，checkbox 7）

- 文件：`deploy/onnx_inference.py`，仅改 line 267 一个实参表达式：
  `voxel_size=float(cfg.dataset.common.voxel_size)` → `voxel_size=cfg.dataset.common.get('voxel_size', None)`。
  去掉 `float(...)` 包装 + 改 `.get(..., None)`，尾随逗号与缩进不变。全文件仅此一处 `voxel_size`（grep 确认），无残留 `float(cfg.dataset.common.voxel_size)`。
- 崩溃机理（BLK-2）：A2 已把 `cfgs/radar/default.yaml:7` 置 `null`，`onnx_inference.py` 默认 `--cfg cfgs/radar/hpenet-ll.yaml` 且 `recursive=True` 级联 → `cfg.dataset.common.voxel_size is None` → `float(None)` 抛 `TypeError`。修复后传给 `preprocess_test` 的就是 `None`。
- 实测证明：`c.get('voxel_size',None)`：`{'voxel_size':None}`→`None`；`{'voxel_size':0.0001}`→`0.0001`；缺键 `{}`→`None`。真实 OmegaConf 合并 `default.yaml(null)+hpenet-ll.yaml` → `None`(NoneType)；`voxel_size:0.02` 的 DictConfig → `0.02`(float)。
- `python3 -m py_compile deploy/onnx_inference.py` → exit 0。
- 兄弟脚本差异（为何只此文件崩）：`deploy/trt_inference.py:263` 与 `deploy/v2_e2e_dump.py:73` 均直接 `voxel_size=cfg.dataset.common.voxel_size`，**无 `float()` 包装**，`None` 可直接透传 → 不崩；trt_inference 另有 `--voxel_size` 覆盖（line 187-188）。A3③ 只修 onnx_inference.py。
- 未动 `onnx_inference.py:175`（`--ckpt` 默认仍指向禁用 run `20260812-201051`）——按任务约束不在本次范围，后续验证任务在命令行显式传正确 ckpt。
- 未改 `preprocess_test` 签名、`deploy/common.py`、trt_inference.py、v2_e2e_dump.py；未跑 ONNX 端到端（F3 负责）；未运行任何 git 命令。

## A3① 完成（2026-09-16，无 git 操作，仅改 `examples/segmentation/main.py` 的 `else` 分支）

### before / after（行号）
- before（main.py:139-140）：
  ```
  139:     else:
  140:         idx_points.append(np.arange(label.shape[0]))
  ```
- after（main.py:139-142）：
  ```
  139:     else:
  140:         idx_part = np.arange(label.shape[0])
  141:         np.random.shuffle(idx_part)
  142:         idx_points.append(idx_part)
  ```

### 验证
- 排列证明：`np.random.seed(0); a=np.arange(10); np.random.shuffle(a)` → `[2 8 4 9 1 6 7 3 0 5]`，`is_identity=False`（确实非恒等）。
- 编译：`python3 -m py_compile examples/segmentation/main.py` → exit 0。
- RNG 恰好消耗一次（单次 `np.random.shuffle`），与体素分支 `main.py:137` 语义一致；点集不变、排列分布与现状同口径。
- 未触碰体素分支、`if voxel_size and downsample` 条件、`main.py:703`（B1）、`deploy/`（A3②/A3③）。

### 依据
- 规格 §3.3：FPS 首中心点恒为输入第 0 点（`sampling_gpu.cu:120-122`）；PLY 按扫描线序写 → 裸 `arange` 会造成系统性 FPS 偏向。§3.2 表 & §6 均把「直接复用 arange 死代码」列为拒绝项。

## A3② 完成：deploy/common.py 无体素分支补 shuffle（2026-09-16）

- 文件：`deploy/common.py`（仅此一处改动）；`preprocess_test` 的 `else` 分支。
- 改动（原 line 71 → 新 line 71-73）：
  - before: `    else:` / `        idx_points.append(np.arange(coord.shape[0]))`
  - after: `    else:` / `        idx_part = np.arange(coord.shape[0])` / `        np.random.shuffle(idx_part)` / `        idx_points.append(idx_part)`
- 语义：与上方 voxel 分支逐条对齐（`idx_part` 局部变量 + 一次 `np.random.shuffle` + append），仅多子云 N 个 → RNG 消耗**恰好一次**；`np.random.seed(100)`（line 62）保持不变，插入的 shuffle **消费同一 seed 流**（预期行为，spec §3.3）。
- 验证：`python3 -m py_compile deploy/common.py` → exit 0；`np.random.seed(100); a=np.arange(10); np.random.shuffle(a)` → `[7 6 1 5 4 2 0 3 9 8]`，非恒等（is_identity=False）→ 证明 `arange` 被真正打乱、FPS 不再固定从 PLY 点 0 起步。
- 边界：两分支入口同为 `np.random.seed(100)`，但调用次数不同（voxel 分支按 count.max() 循环 N 次，else 分支 1 次），voxel_size 非 null 时行为零变化；`if voxel_size is not None:` 条件与 voxel 分支逻辑未动。
- 未运行任何 git 命令，未跑部署管线；`deploy/onnx_inference.py` 等其他 deploy 文件未触碰。

## C1+C2 完成：CPP_trt4 pipeline.cpp 去体素化（2026-09-16，checkbox 9/10）

- 文件：`deploy/CPP_trt4/src/pipeline.cpp`，仅此一处改动；`voxel_size` 参数/成员全链路保留（C5 任务），`voxelizer.h` include 保留（C4 任务），未删任何文件，未动 CMakeLists.txt、trt_inference_wrapper.cpp。
- C1（process_pointcloud，原 :111-112）：删 `Voxelizer::voxelize(coord.data(), num_points, voxel_size_, seed_)` → 本地构造 `std::vector<int> idx(num_points)` + `std::iota` + `NumpyMT19937 rng(seed_)` + `rng.shuffle(idx.data(), num_points)` + `std::vector<std::vector<int>> idx_points{std::move(idx)}`。原 :117/:132 的 `vox.idx_points` → `idx_points`。
- C2（process_file，原 :305-308）：同上（用 `pc.num_points`），`nvtxRangePushA("VOXELIZE")`/`nvtxRangePop()` 一对**整体删除**（F4 判据）。原 :313/:327/:330 的 `vox.idx_points` → `idx_points`；`:327` 改写为 `size_t count_subcloud = idx_points.size();  // 去体素化后恒为 1`。`:474`（现 :481）`result.count_subcloud = count_subcloud;` 未动。
- include 补齐（spec §4 item 2）：`#include <numeric>`（`std::iota`，选 iota 而非 for 循环——照抄 spec §3.3 片段、与 main.cpp:10 已有用法一致）+ `#include "random_util.h"`（头文件在 `include/`，pipeline.cpp 原 include 块无）。
- 类型口径：`idx_points`/`idx_part` 保持 `int`（消费端 `preprocess_subcloud(const int*)`）；`idx_staging_`/`d_idx` 仍 `int64_t`，未动。
- RNG 语义确认（voxelizer.cu:161-171）：单实例 `NumpyMT19937 rng(seed)`，RNG 只被 shuffle 消耗；现役 shuffle 长度是 `M`（体素数），`voxel_size=0.0001f` 下 M==N（基线 10 文件均 subcloud=1）→ N 长度 shuffle 是今日行为的精确复现。
- `NumpyMT19937` ctor 为 `explicit NumpyMT19937(uint32_t seed=100)`；`seed_` 为 `int` 成员（pipeline.h:107）→ 直接初始化 `NumpyMT19937 rng(seed_)` 不受 explicit 影响，int→uint32_t 隐式转换无碍。
- 残留扫描（grep `\bvox\b|VoxelizeResult|Voxelizer|VOXELIZE`）：仅 `pipeline.cpp:6` 文件头流程注释 `// 3. Voxelizer::voxelize → idx_points (子云列表)` 命中——属 spec C4「过期注释，可选清理」清单，本任务不动。**代码级 `vox`/`VoxelizeResult` 引用 = 0。**
- 编译：增量 `make -j`（build/ 缓存已带 `-DCUDNN_LIB=...`，基线 sha256 0b52d744... 确认未漂移）→ **exit 0**，仅重编 `pipeline.cpp.o` + relink；新二进制 sha256 见构建产物（未做 clean rebuild——spec §4 item 4 的 clean rebuild 要求属 C3/C4 删文件后的任务，本任务任务书明确说增量 make 足够）。
- 未运行任何 git 命令；未运行 CLI 推理（F4 延迟/精度验收由后续任务负责）。

## A4 验收：方案 A（voxel_size: null）acc 对拍 + 1-epoch 训练（2026-09-16，GPU 3/4，无 git 操作，源文件零改动，checkbox 8）

### 判定总览
- **Part1 test（spec §5.2 val/test 行）：PASS** —— ΔOA = **−0.07pp**（判据 < 0.3pp）；所有聚合指标（OA/mACC/mIoU/mP/mR/iou×2）Δ 绝对值 ≤ 0.16pp。两次复跑 2dp 口径逐位一致（仅 rec_invalid 末位 83.52/83.51），新配置下确定性水平与基线相同。
- **Part1 崩溃检查：无 TypeError** —— A1（s3disRadar.py novx tag）修复得到直接证明；两次 mode=test 全流程跑通。
- **Part2 train（spec §5.2 train 行）：PASS** —— train_miou +0.018pp、train_macc +0.034pp（判据 < 1.0pp）；train_loss 0.7116 vs 0.7112，无 NaN/Inf/发散。

### Part1 命令（与 P1 基线逐字符同命令；GPU 3）
```
source /home/wangpeng/miniforge3/etc/profile.d/conda.sh && conda activate hpenet && CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml wandb.use_wandb=False mode=test --pretrained_path log/radar/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV/checkpoint/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV_ckpt_best.pth
```
生效 seed：`main.py:606 set_random_seed(0)`（test() 内，data loop 前），与基线同。

### Part1 结果（58 文件全量 test 集，2dp 口径）
| metric | 基线 0.0001（P1） | 新 null（run1） | Δ |
|---|---|---|---|
| OA | 92.05 | 91.98 | **−0.07** |
| mACC | 88.50 | 88.58 | +0.08 |
| mIoU | 77.02 | 76.92 | −0.10 |
| mP | 84.55 | 84.39 | −0.16 |
| mR | 88.50 | 88.58 | +0.08 |
| iou_valid | 90.79 | 90.71 | −0.08 |
| iou_invalid | 63.26 | 63.13 | −0.13 |
| prec_valid | 96.60 | 96.65 | +0.05 |
| prec_invalid | 72.51 | 72.12 | −0.39 |
| rec_valid | 93.79 | 93.65 | −0.14 |
| rec_invalid | 83.21 | 83.52 | +0.31 |

- run2 与 run1 完全一致（仅 rec_invalid 83.51）→ prec/rec_invalid 的 ±0.3~0.4pp 偏移**可复现、非抖动**：与 spec §3.4「非 bit 等价、统计等价」预测一致（少数类边界点翻转，prec/rec 反向移动，谐波 iou_invalid 仅 −0.13pp）。
- 复跑耗时：run1 37s、run2 44s（GPU 3 / L20）。
- 框架行为复现：新增 `...20260916-135704-*` / `...20260916-135805-*` 两个 test log + `_test.csv` 追加 2 行（共 4 行：2 基线 + 2 新），pinned 训练 CSV/日志/ckpt 未动。

### Part2 命令（GPU 4；seed=47 依据：pinned run 日志头部 `seed: 47`，cfg.seed:null → 运行期随机数，必须显式钉回同 seed 才同口径）
```
source /home/wangpeng/miniforge3/etc/profile.d/conda.sh && conda activate hpenet && CUDA_VISIBLE_DEVICES=4 python examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml wandb.use_wandb=False seed=47 epochs=1
```
新 run 目录：`log/radar/radar-train-hpenet-ll-ngpus1-20260916-135913-82KZRLCQkMJeXBCkrFpQ7V`（ckpt_best/latest 均保存）；耗时 137s（epoch1 train+val ≈109s）。

### Part2 epoch-1 逐字对比
- 基线（pinned run 日志 L892-896）：
  ```
  [09/07 17:06:44] RadarClassi INFO: Find a better ckpt @E1, val_miou 39.45 val_macc 60.44, val_oa 62.46, val_mp 56.08, val_mr 60.44
  mious: [58.45 20.45]
  precisions: [88.06 24.11]
  recalls: [63.49 57.4 ]
  [09/07 17:06:44] RadarClassi INFO: Epoch 1 LR 0.000001 train_miou 35.46, val_miou 39.45, best val miou 39.45, val_mp 56.08, val_mr 60.44
  ```
- 新 run（L892-896）：
  ```
  [09/16 14:00:57] RadarClassi INFO: Find a better ckpt @E1, val_miou 38.50 val_macc 60.06, val_oa 61.08, val_mp 55.78, val_mr 60.06
  mious: [56.84 20.15]
  precisions: [88.04 23.52]
  recalls: [61.6  58.51]
  [09/16 14:00:57] RadarClassi INFO: Epoch 1 LR 0.000001 train_miou 35.48, val_miou 38.50, best val miou 38.50, val_mp 55.78, val_mr 60.06
  ```

### Part2 TB scalars @E1（events 文件精确值）
| metric | 基线 0.0001 | 新 null | Δ |
|---|---|---|---|
| train_loss | 0.712205 | 0.711613 | −0.00059 |
| train_miou | 35.4648 | 35.4832 | +0.018 |
| train_macc | 56.6081 | 56.6416 | +0.034 |
| train_mp | 53.7102 | 53.7284 | +0.018 |
| train_mr | 56.6081 | 56.6416 | +0.034 |
| val_miou | 39.4516 | 38.4979 | −0.954 |
| val_oa | 62.4631 | 61.0810 | −1.382 |
| val_macc | 60.4418 | 60.0550 | −0.387 |

- LR 双跑一致（warmup 首步 1e-6）。train 判据全过；**val@E1 偏低属 warmup 首步 + RNG 流偏移的正常噪声**（收敛后 test acc −0.07pp 才是核心判据），不构成异常、未记 issue。

### 最终状态核验（运行后）
- `cfgs/radar/default.yaml:7` 仍为 `voxel_size: null #...`（未改）。
- 5 处改动锚点逐行核验在位：A1（s3disRadar.py:73 novx tag）、A2（default.yaml:7 null）、A3①（main.py:139-142 shuffle）、A3②（common.py:69-73 shuffle）、A3③（onnx_inference.py:267 `.get('voxel_size', None)`）。
- 未运行任何 git 命令；未触碰 deploy/CPP_trt4、deploy/trt_inference.py、deploy/onnx_inference.py（F3 范围）。

## C5 (checkbox 11) — voxel_size 全链路移除 — 2026-09-16
- 站点核实与 spec C5 行一致；实际 main.cpp 有两个解析分支（`--key=value` 的 :123 与 `--key value` 的 :143，任务书行号标注 143 为构造传参有偏差，构造传参实际在 :185 附近），三处全删。
- `pipeline.cpp` 原 :113/:312 的两条 `// voxel_size_ = 0.0001f ...` 注释随成员删除一并移除（引用已不存在成员），其后的 FPS/shuffle 理由注释保留。
- `pipeline.h:44` seed 文档措辞 `体素化随机种子` → `随机种子 (子云 Fisher-Yates shuffle)`（体素化概念已删，原文案失效；spec C4 已预告该行文案会陈旧）。
- `main.cpp` 注释块（旧 CLIConfig）内的 `// float voxel_size = 0.02f;` 一并删除（任务书明确列为站点）。
- 剩余 `voxel_size` 命中仅限 C3/C4 删除清单文件（voxelizer/fnv_hash .cu/.h）与 `test.c:132` 注释（属 C6 范畴，wrapper 的 0.02f 未动）。
- 编译：增量 `make -j`（cmake 带 -DCUDNN_LIB 重配）exit 0；`voxelizer.cu` 仍在编译（C3 才移除）。
- `--help` 冒烟（带 pip cuDNN LD_LIBRARY_PATH）exit 0，help 输出无 --voxel_size；`--voxel_size=0.02` 直接运行 exit 1（Unknown argument）。
- 新二进制 sha256 974505156c9a1b762414a8e1e2d4d4db686f79463ed4421a728bfb39a4d5e469，3,778,016 B。
- 与备份 tarball 逐字节 cmp：trt_inference_wrapper.cpp / voxelizer.{cu,h} / fnv_hash.{cu,h} / CMakeLists.txt 全部 UNTOUCHED。

## C3 (checkbox 13) — CPP_trt4/CMakeLists.txt 移除已删源文件 — 2026-09-16
- 取证（编辑前 grep `voxelizer|fnv_hash`，排除 build/ 与 C4 所属源文件）：构建输入中**仅** `CMakeLists.txt:64 src/voxelizer.cu` 与 `:65 src/fnv_hash.cu` 两处；`cmake/FindTensorRT.cmake` **零命中**；无其他 CMakeLists/.cmake/脚本引用删除目标。
- CMakeLists 结构枚举（全 109 行读完）：唯一的源列表 = `add_executable(${TARGET_NAME} ...)`(l51-68)，共 16 个源（无第二个 target、无重复列表）；**无 header 列表**（头文件仅靠 l69 `target_include_directories(... include ...)` 的目录暴露，故 `.h` 不在 CMake 内需改）。l31-48 的 googletest 块全注释（`add_subdirectory(tests)` 未启用）。`add_subdirectory(../trt_plugins)`(l73) 属插件子工程，与本删除无关。
- before hunk（原 l63-68）：
  ```
  63:     src/cuda_utils.cu
  64:     src/voxelizer.cu
  65:     src/fnv_hash.cu
  66:     src/scatter_mean.cu
  67:     src/trim_transpose.cu
  68: )
  ```
  after hunk（新 l63-66）：
  ```
  63:     src/cuda_utils.cu
  64:     src/scatter_mean.cu
  65:     src/trim_transpose.cu
  66: )
  ```
  仅删除 2 行，行序/缩进/其余源全保留；文件 109→107 行。
- **`src/scatter_mean.cu` 明确保留**（新 l64）—— 按 spec「保留文件 ... 保留不影响编译/正确性」；B3(checkbox 18) 令 `launch_scatter_mean_kernel` 变为未被调用，此后该 .cu 即死代码，但本任务不删。
- 合法性核验（纯静态，未跑 cmake）：全文 `(`=58 `)`=58 平衡；`add_executable` 参数表括号配对、无悬挂续行（`src/trim_transpose.cu` 后紧跟 `)`）。
- 编辑后 grep：`CMakeLists.txt`/`cmake/` 中 `voxelizer|fnv_hash` **零命中**（exit 1）。build/ 内残留（`link.txt`、`compile_commands.json`、`Makefile`、`DependInfo.cmake`、`*.o.d` 等）属 CMake 生成产物，**仍引用 `src/voxelizer.cu.o` / `src/fnv_hash.cu.o`** —— 再次确认 C7 必须 **clean rebuild**，增量 `make` 会因找不到已删源/陈旧 link.txt 失败。`src/fnv_hash.cu:1 #include "fnv_hash.h"`、`src/pipeline.cpp:33 #include "voxelizer.h"`（C4 负责删）、`src/trim_transpose.cu:14` 注释提及均为非 CMake 引用，不在本任务范围。
- 未运行任何 git 命令；未跑 cmake/make；未删任何文件；未动 plans/其他 CMakeLists/其他项目。

## C6 (checkbox 12) — 修复 BLK-1 静默 misbinding — 2026-09-16

### 构造签名（逐行核实，verbatim，`include/pipeline.h`）
```
44:    InferencePipeline(
45:        const std::string& engine_path,
46:        const std::string& stats_json_path,
47:        TrLogger& logger,
48:        int min_n = 1024,
49:        int max_n = 10000,
50:        int seed = 100);
```
参数顺序：`engine_path, stats_json_path, logger, min_n, max_n, seed`（共 6 个，无 `voxel_size`）。

### 调用点逐一映射（`src/trt_inference_wrapper.cpp:192-194`）
| # | 实参（fix 前） | 形参 | 绑定 |
|---|---|---|---|
| 1 | `std::string(onnx_path)` | `engine_path` | ✅ |
| 2 | `std::string(stats_json_path)` | `stats_json_path` | ✅ |
| 3 | `glogger` | `logger` | ✅ |
| 4 | `2024` | `min_n` | ✅ |
| 5 | `10000` | `max_n` | ✅ |
| 6 | `0.02f` → `100` | `seed` | 🔴 原为 misbind |

- BLK-1 证实：C5 删掉 `voxel_size` 形参后，`0.02f` 落到第 6 位 `seed`（int）；float→int 隐式转换截断为 **0**，仅可能触发 narrowing 警告、无编译错误 → 静默把 seed 从默认 100 变 0。

### before / after
- before（:192-194）:
  ```
  192:            handle->cpp_pipeline = std::make_unique<InferencePipeline>(
  193:                std::string(onnx_path), std::string(stats_json_path), glogger,
  194:                2024, 10000, 0.02f);
  ```
- after（:192-194）:
  ```
  192:            handle->cpp_pipeline = std::make_unique<InferencePipeline>(
  193:                std::string(onnx_path), std::string(stats_json_path), glogger,
  194:                2024, 10000, 100);
  ```
  仅第 6 实参 `0.02f` → `100`，未重排、未重格式化。

### seed=100 依据
- `src/main.cpp:35`：`int seed = 100;`（CLIConfig 默认，CLI 构造共用同一语义）。
- `include/random_util.h:21`：`explicit NumpyMT19937(uint32_t seed = 100);`。
- C-API 路径历史默认即 100；改为显式 `100` 恢复 C5 前的 seed 语义。

### 核验
- `grep -n "voxel_size" deploy/CPP_trt4/src/trt_inference_wrapper.cpp` → **0 hits**（exit 1），参数已彻底移除。
- 活跃构造点唯一：`grep -n "make_unique<InferencePipeline>"` → `:192`（`:149` 为注释块内），无第二处陈旧实参顺序。
- **未运行任何 build**（C3/C4 并发改 CMake/删源，编译门属 C7）；未跑 git；未触碰其他源文件。

### 语义提醒（spec §4）
C-API 旧 `0.02f` 曾是**真实体素化**（250/339 文件多子云），CLI 侧 `0.0001f` 才是 no-op。去体素化对 C-API 行为影响远大于 CLI → F4 对 C-API 必须做**精度级**（非预测级）对比。

## C4 (checkbox 14) — 删除 voxelizer/fnv_hash 四文件 + 清过期注释 — 2026-09-16

### 删除前取证（inventory）
- 四文件（大小）：`src/voxelizer.cu` 7,965 B / `include/voxelizer.h` 2,961 B / `src/fnv_hash.cu` 2,733 B / `include/fnv_hash.h` 520 B。
- CMakeLists.txt 状态（C3 已落地）：grep `voxelizer|fnv_hash` **零命中**；`src/scatter_mean.cu` 在新 l64（保留）；16→14 个源。
- 全树 grep `voxelizer\|fnv_hash` 命中分类（删除前）：
  - src/include 内：4 个待删文件自身；`pipeline.cpp:33` 真实 `#include "voxelizer.h"`；`pipeline.h:13` 注释掉的 `//#include "voxelizer.h"`；`pipeline.cpp:6` 文件头 Step 3 流程注释；`trim_transpose.cu:14` sync 设计注释（`scatter/fnv_hash/voxelizer 的 launch 才加`）。
  - `src/test.c` / `src/test_rpc.c`：**零命中**（不在 CMake 构建内，无需处理）；无 `test_*.cu` 文件。
  - `include/tinyply.h` 的 `hash_fnv1a`（FNV-1a 字符串哈希）是第三方 tinyply 自有代码，**与被删 `fnv_hash.h` 无关**，未动。
  - build/ 大量命中（Makefile/link.txt/compile_commands.json/*.o.d/*.o/二进制等）→ 全部为 CMake 生成产物（分类 (a) 非构建输入），C7 clean rebuild 自愈；印证「只 make 必失败」结论。

### 执行
- 删除 4 文件（`rm`）；未动 `random_util.{h,cpp}`、`scatter_mean.{cu,h}`（存活核验：14 个文件全在，见上）。
- `pipeline.cpp:33` 删 `#include "voxelizer.h"`（include 块现以 `trim_transpose.h` 收尾）。
- `pipeline.cpp:6` 头注释 Step 3：`Voxelizer::voxelize → idx_points (子云列表)` → `去体素化: std::iota + NumpyMT19937 Fisher-Yates shuffle → idx_points (单子云列表)`（与 C1/C2 落地的 :110 Step 3 注释一致）。
- `pipeline.cpp:15` Step 5（launch_scatter_mean_kernel → GPU 合并）**保留不改**：scatter 内核调用现役（B3/checkbox 18 才移除），描述仍真实。
- Step 6 注释（:238/:444「下载合并后的 logits」）**保留不改**：当前仍真实（B3 才改区域），非 plainly false，任务书允许留待 B3。
- `include/pipeline.h:13` 注释掉的 `//#include "voxelizer.h"` 删除（过期注释清理；非 C5 的参数列表，未越权）。
- `src/trim_transpose.cu:14`：`scatter/fnv_hash/voxelizer 的 launch 才加` → `scatter_mean 的 launch 才加`（fnv_hash/voxelizer launch 已不存在，最小化修正）。
- `include/pipeline.h:44` 核验：`随机种子 (子云 Fisher-Yates shuffle)`（C5 已修），**未再编辑**。

### 删除后 sweep
- `grep -rn "voxelizer\|fnv_hash" deploy/CPP_trt4/src deploy/CPP_trt4/include` → **零命中（exit 1）** ✅
- 全树 `grep -rln` 剩余命中**仅 build/**（生成产物，(a) 类，含陈旧 `fnv_hash.cu.o`/`voxelizer.cu.o` 目标文件与旧 link.txt）→ C7 必须删 build/ 后 clean rebuild。
- 未运行任何 build / git；未编辑 CMakeLists.txt（C3）、trt_inference_wrapper.cpp（C6）、main.cpp、types.h、plans、其他项目。

## C7 (checkbox 15) — C 侧验收门：clean rebuild + CLI + nsys + C-API（2026-09-16，GPU 5 / L20，无 git 操作，源文件零改动）

### A. clean rebuild（rm -rf build 后全新构建）
- `rm -rf build && mkdir build` exit 0；`cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89" -DCMAKE_BUILD_TYPE=Release -DCUDNN_LIB=/home/wangpeng/miniforge3/envs/hpenet/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8` exit 0；`make -j` **exit 0**。
- 新二进制 `build/hpenet_trt_infer` **1,596,440 B**；sha256 `e21f435f26e6c0e32780a6efbdbc1fa1f845bd4a1164187538ac81d92445199a`；md5 `4bc48659a3bf3c289794066aade0886e`（not stripped；旧基线 3,777,896 B 是 CPP_trt3 拷来的旧 build 产物，口径不同）。
- 新 link.txt grep `voxelizer|fnv_hash` = **0 命中** → 陈旧引用问题根治（重建前旧 link.txt 确实仍引用 voxelizer.cu.o/fnv_hash.cu.o，取证留档）。
- 编译警告：仅 `cuda_utils.h:26 unused template param T` ×4（无害、与基线同源）；**无 float→int narrowing 警告** → BLK-1 修复（wrapper :194 `2024,10000,100`）无侧漏。
- `--help` exit 0、help 无 --voxel_size；`--voxel_size=0.02` exit 1 "Unknown argument"。plugins 静态库 `build/hpenet_plugins_build/libhpenet_plugins.a` 1,429,650 B。

### B. CLI（--num_files 10，GPU 5，与 P2 基线同引擎/同 stats/同 data_dir）
- 文件集相同：0000068..0000077，全部 subcloud=1。两次运行逐文件 acc 完全一致（推理确定性）。
- **acc 与基线非逐位一致**（任务书预期逐位一致，实测不是）：逐文件 0.9282/0.9297/0.9358/0.9288/0.9477/0.9347/0.9456/0.9446/0.9318/0.9450，**Mean 0.9372**（基线 0.9358）→ **Δmean +0.14pp**；逐文件 Δ 范围 −1.14pp(0068) ~ +1.76pp(0076)。
- **根因已实证（非 hand-wave，见下取证链）**：旧 voxelizer.cu:164-171 shuffle 的对象是 **FNV-hash 排序后的点序 idx_s**，新 pipeline.cpp:111-114 shuffle 的对象是 **扫描序 iota**。RNG 消耗序列完全相同（同 seed=100、同 N 长 Fisher-Yates），但被打乱的输入数组不同（idx_s ≠ identity，FNV 序是伪随机排列）→ FPS 首中心点不同 → 预测必然不同。spec §3.3/§3.4 本来就写「非 bit 等价 → 必须按 acc 验收」，Δacc<0.3pp 判据下 **0.14pp PASS**（该判据最终由 F4 裁定）。任务书「逐位一致」的前提（误以为旧 shuffle 输入是 identity）不成立。
- 取证链（全部 /tmp/opencode 证据）：
  1. C4 删掉的 voxelizer.cu 与 CPP_trt3/src/voxelizer.cu 同大小 7,965 B（基线二进制的体素化即此代码）。
  2. `shuffle_forensics.py`：0000068.ply N=3707、M=3707（无体素合并）；idx_s≠identity；旧路径 FPS 首点 idx=3550、新路径 idx=1187，首个分歧位置 0。
  3. `rng_crosscheck.cpp`：独立编译 random_util.cpp 的 C++ NumpyMT19937(100).shuffle(0..31) 与 numpy RandomState(100).shuffle 逐元素一致（[13 28 1 26 5 ...]）→ Python 模拟忠实于 C++ 路径。
- 引擎磁盘文件全程未变：`deploy/hpenet_v2_fp32.engine` 14,679,492 B / 2026-09-07 20:42；stats `deploy/CPP_trt/stats_feat5.json` 296 B。前后对比用的是同一 engine → before/after 有效。

### C. nsys 剖析（同基线方法：nsys profile --trace=cuda,nvtx + nsys export --type sqlite；注意 nsys 2022.4 不自动产 .sqlite，需 `nsys export --type sqlite`，基线 sqlite 的 EXPORT_META_DATA 证实同法）
- 新 profile `/tmp/opencode/cpp_trt4_after_voxel_removal.{nsys-rep,sqlite}`（768,667 B / 1,822,720 B）。
- **VOXELIZE 全表 0 事件**（基线 10 个）→ 删除确认 ✅。
- analyze_latency.py（NUM_FILES=10，与基线同口径）：端到端 **9.476 ms/文件**（基线 10.455，**−9.4%**）；部署口径 **4.177 ms/帧**（基线 4.247，**−1.6%**）；PLY_LOAD 5.299（6.208）/ SUBCLOUD_LOOP 3.402（2.460）/ TAIL 0.762（1.312）/ COORD_SHIFT 0.006 / ARGMAX_ACC 0.007；GPU kernels **2.619 ms/文件**（基线 2.686，−0.067，即删掉的 hash+sort 内核）。
- 注意：SUBCLOUD_LOOP+TAIL 之和反而 +0.39（4.165 vs 3.772 每文件），逐文件两段反相关（CPU enqueue 快则 TAIL 等同步长），属调度噪声；spec CLI 行的「端到端 ≥5%」口径达成（−9.4%，但受 PLY_LOAD IO 噪声影响大），「部署口径 ≥5%」本单次 −1.6% 未达——**延迟终判属 F4**（任务书明示），此处仅如实记录单次数据。

### D. C-API 入口：可行，已实测（acc 级 + 逐位自洽）
- 驱动方式取证：`src/test.c`（canonical：create → 每文件 trt_pipeline_process_file → destroy）与 `src/test_rpc.c`（SimCdi 布局 + trt_ai_infer_and_update 冒烟，合成数据无 GT）**均不在 CMake 构建内**；兄弟工程无 python ctypes 包装、无现成 acc 对拍驱动 → 需自建。
- 自建驱动 `/tmp/opencode/capi_driver.cpp`（临时 QA 件，不入仓库）：链接 = 复用新 link.txt、把 `main.cpp.o` 换成驱动 .o、`-o /tmp/opencode/capi_driver`（链接 exit 0，二进制 1,600,544 B）。
- 覆盖三个入口 × 10 文件（0000068..0000077，同 ti10 选择 n_total=339/test_start=67）：`trt_pipeline_process_file`（写 result*.ply）、`trt_pipeline_process_inmemory`（pred 落 .inmem.pred.bin）、`trt_ai_infer_and_update`（SimCdi 布局、**调用前备份 valid=GT** 落 .gt.valid.bin、调用后落 .cdi.valid.bin）。driver exit 0；各入口 latency 2.6~3.3 ms/文件（首个 process_file 75ms 为引擎上下文创建）。
- **对拍结果（capi_compare.py）**：
  - 四路预测源（CLI dump pred.bin / process_file 输出 PLY label / inmemory pred / infer_and_update 更新后 valid）**逐位一致**（10 文件全等）→ CLI 与 C-API「两入口结果自洽」✅（§5.2 构建行判据）。
  - GT 备份与原始 GT 一致（断言通过）→ infer_and_update 原地覆盖 valid 属实，备份法可对拍（spec §5.2 警示已验证）。
  - 每入口逐文件 acc 完全相同，**mean acc = 0.9372**（= CLI 同一次口径）。
- 局限声明（诚实口径）：C-API 旧行为（voxel 0.02f 真体素化、250/339 文件多子云）**无基线 acc 记录**，旧二进制已被 clean rebuild 覆盖、CPP_trt3 二进制禁跑 → 无法做「C-API 去体素化前后」的直接 acc 对拍；本次证明的是「新 C-API 入口跑通 + 与 CLI 逐位自洽 + acc 0.9372」。与「去体素化后 Python 参考 acc <0.3pp」的 C-API 终判需 F4（需 Python 参考 acc，本任务禁跑 Python 管线）。

### QA 产物清单（全部保留在 /tmp/opencode/，未清理，供 F4 复核）
- 证据：`capi_driver.cpp/.o/capi_driver`、`capi_compare.py`、`shuffle_forensics.py`、`rng_crosscheck.cpp/rng_crosscheck`、`cpp_trt4_clean_build.log`、`cpp_trt4_cli_run1.log`、`cpp_trt4_cli_run2.log`、`capi_driver.log`、`cli_dump/`(20)、`capi_dump/`(30)、`capi_out/`(10)、`cpp_trt4_after_voxel_removal.{nsys-rep,sqlite}`。
- 仓库内零新增文件（build/ 为 gitignored 重建产物）；未跑任何 git 命令；GPUs 0-7 除 6/7（他人任务）外全部还原空闲。

## B1 落地（2026-09-16，checkbox 16，仅改 `examples/segmentation/main.py`，无 git 操作）

### 行号修正（任务书提前 2 行）
- 任务书称 scatter 在 `:703`、`pred = all_logits.argmax` 在 `:707`；**实测二者分别在 `:705` / `:709`**（`:703` 是注释行 `# average merge overlapped multi voxels logits to original point set`）。已按实际 `:705` 编辑。

### before / after（实际行号）
- before（main.py:704-705）：
  ```
  704:            idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
  705:            all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
  ```
- after（main.py:704-708，`if not nearest_neighbor:` 分支）：
  ```
  704:            idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
  705:            out = torch.empty_like(all_logits)
  706:            out[idx_points] = all_logits
  707:            all_logits = out
  ```
- `:709 pred = all_logits.argmax(dim=1)` 确认消费名为 `all_logits` 的变量（未改名）→ **回绑 `all_logits = out` 已在位**（B1 唯一红线）。
- `else`（nearest_neighbor）分支与 `:703` 注释均未动；`validate()` 的 `:554 scatter(...)` 不属 B1 范围，保留。

### 消费链核实
- `all_logits` 来源：`:697 append(logits)` → `:698 torch.cat(dim=0)` → `:700 transpose(1,2).reshape(-1, cfg.num_classes)`。radar 单子云下形状 `(N, num_classes)`，dtype = 模型输出 = **float32**（GPU）。
- `idx_points`：`load_data`（`:139-142`，A3①）产出 `np.arange(N)` 经 `np.random.shuffle` 的**均匀随机排列**，列表长度 1；`:704 np.hstack` → shape `(N,)`，dtype `int64`（numpy 默认），`.cuda()`。
- `:709` 唯一后续消费者；`:711 cm.update(pred, label)`、`:714` 可视化（注释）均用 pred/all_logits。

### 真实数据排列完整性证明（`/tmp/opencode/b1_perm_check.py`）
- 直接 `sys.path.insert(examples/segmentation)` 后 `import main`，复用仓库自身 `load_data` + 真实 PLY（`data/RadarClassi/radarfullwl/raw`，339 文件），复刻 test() 口径（先 `np.random.seed(0)`，再逐文件 load_data）：
  - `files_checked=339  bad=0  N_range=[3005,7837]`；每个文件断言 `len(idx_points)==1`、`label.shape[0]==N`、`np.unique(idx).size==N`、`idx.min()==0`、`idx.max()==N-1`。**排列完整、无重复无缺号**。
- 语义等价（真实 idx，CPU torch_scatter）：`scatter(logits, idx, reduce='mean')` 与 `out=empty_like(logits); out[idx]=logits` → `torch.equal == True`（bit 级相等，非仅 allclose）。因 count 恒为 1，均值为 no-op，`除以 1` 不损精度。
- **未初始化行证明**：`empty_like` 结果与 `zeros_like` 结果 `torch.equal == True` → 全行被写、无垃圾行可被 argmax 读到。
- dtype/shape：`empty_like` 得 `(N, 2)` / `torch.float32`（真实为 `(N, num_classes)`）；device 由 `empty_like` 自动继承 GPU。
- 确定性/数值：scatter mean 与 B1 均为「每 target 恰写一次」，排列下均与源写入顺序无关 → **无预期数值差异**（仅 float 归约确定性层面；本已验证 bit 级相等）。

### 编译
- `python3 -m py_compile examples/segmentation/main.py` → **exit 0**。
- 未跑训练/测试（验收由 checkbox 19 / B4 负责）；未碰 `.omo/plans/`；未运行任何 git 命令；未动 `deploy/*`（B2 并发）。

## B2 落地（2026-09-16，checkbox 17，仅改 `deploy/trt_inference.py` + `deploy/onnx_inference.py`，无 git 操作）

### 五站点 before / after（实际行号；旧行号 → 新行号）
统一替换（引号风格随原文件：trt 用 `"`、onnx 用 `'`；新代码两文件逐字相同）：

- `deploy/trt_inference.py:88`（`infer_one_cloud_trt`）
  - before：`    merged = scatter(all_logits_cat, idx_flat, dim=0, reduce="mean")`
  - after（`:88-89`）：
    ```
    88:    merged = torch.empty_like(all_logits_cat)
    89:    merged[idx_flat] = all_logits_cat
    ```
- `deploy/trt_inference.py:113`（`infer_one_cloud_onnx`）→ after `:114-115`（同上两行，缩进 4 空格）
- `deploy/trt_inference.py:141`（`infer_one_cloud_pytorch`）→ after `:143-144`（同上两行）
- `deploy/onnx_inference.py:120`（`infer_one_cloud_onnx`）
  - before：`    merged = scatter(all_logits_cat, idx_flat, dim=0, reduce='mean')  # (N_orig, 2)`
  - after（`:120-121`）：`merged = torch.empty_like(all_logits_cat)  # (N_orig, 2)` + `merged[idx_flat] = all_logits_cat`（**尾注释 `# (N_orig, 2)` 已按任务书保留**）
- `deploy/onnx_inference.py:162`（`infer_one_cloud_pytorch`）→ after `:163-164`

局部变量名五站点**逐行核实**均为 `merged` / `all_logits_cat` / `idx_flat`，无偏差。
`torch` 在两文件均已在模块顶部 import（`trt_inference.py:26`、`onnx_inference.py:21`）。

### 行号漂移（重要，供 F3/后续任务）
`onnx_inference.py` 因两站点各 +1 行（1 行 → 2 行），**全文 +2 行**：A3③ 的 `cfg.dataset.common.get('voxel_size', None)` 由 `:267` 移到 **`:269`**（内容未动，已核实）；B2 第二站点原 `:162` → `:163`。`trt_inference.py` 因三站点各 +1 行，**全文 +3 行**（原 `:88` 不变、原 `:113` → `:114`、原 `:141` → `:143`）。后续任务引用行号时请以本节点为准。

### 真实数据排列完整性证明（`/tmp/b2_perm_proof.py`，真实 PLY `data/RadarClassi/radarfullwl/raw/0000001.ply`）
- 目标口径 `voxel_size=None`（A2 已把默认 0.0001 改 null）：`N_orig=5119`、`n_subclouds=1`、`idx_flat` len **5119**、unique **5119**、min/max **0/5118**、`np.sort(idx_flat) == np.arange(N)` **True**、每点 multiplicity **1**、子云尺寸之和 == N。
  → **`idx_flat` 是 `0..N-1` 的完整排列（无重复、无缺号）**，`empty_like` 每一行都会被恰好写一次，**无未初始化行**。
- 解析证明（对应代码）：`common.py:70-73`（A3②）在 `voxel_size is None` 时 `idx_part = np.arange(N)` 后 `np.random.shuffle` → 均匀随机排列；`idx_points` 长 1；`idx_flat = torch.cat(all_idx)` 即该排列。旧体素路径在 `count.max()==1` 时 `idx_sort` 亦为 `0..N-1` 排列（N 点 N 格），与此一致。

### 数值等价（bit 级，非仅 allclose）
- 真实 idx + 真实 `all_logits_cat`：`scatter(...,'mean')` == `out=empty_like(cat); out[idx]=cat` → `torch.equal == True`（count 恒 1，均值除以 1，无精度损失）。
- **实际编辑后的函数**（stub session，真实数据）：`infer_one_cloud_trt` / `infer_one_cloud_onnx` / `infer_one_cloud_pytorch`(trt) / `infer_one_cloud_pytorch`(onnx) **四个 helper 输出均与 scatter-mean 参考 `torch.equal == True`**；`_pytorch` 两站点用 `.cuda()` 恒等 monkeypatch 在 **CPU** 上跑通（未占用任何 GPU）。
- **重排确实被保留（防静默 acc 的核心）**：`merged` == 「原始点序写入」**True**；`merged` == 「子云序（即丢掉重排后的错序）」**False**。证明本改动**不是** B1/B3 驳回的「跳过 scatter」。
- 反例（合成、非目标口径）：含重复 index 时 mean 形式与 write 形式数值不同 → 说明 target 配置下「count==1」是等价前提；代码注释亦已提示该假设。

### 类型/形状（代表站点）
`all_logits_cat.shape=(5119, 2)`、`dtype=torch.float32`、`is_contiguous=False`（来自 `transpose(0,1)`）。`torch.empty_like` 保留 dtype/shape/device/strides → `merged` 同为 `(5119,2)/float32/非连续`，`.argmax(dim=1)` 正常消费。

### 检查
- `grep -n "scatter(" deploy/trt_inference.py deploy/onnx_inference.py` → **零命中（exit 1）**；`from torch_scatter import scatter`（`trt:87/113/142`、`onnx:119/162`）**有意保留**（未使用但不匹配 sweep，保持最小 diff；spec B2 只规定调用点替换）。
- `python3 -m py_compile deploy/trt_inference.py deploy/onnx_inference.py` → **exit 0**。
- 未跑部署脚本端到端（F3 负责）；未跑任何 git 命令；未碰 `.omo/plans/omitting-voxelize.md`（checkbox 由 orchestrator 维护）；未动 `deploy/common.py`、`deploy/onnx_inference.py:269`(A3③)、`deploy/v2_e2e_dump.py`、`examples/segmentation/main.py`(B1)、`deploy/CPP_trt4`(B3)。
- 遗留（**非本轮改动，供 F5 决定**）：`from torch_scatter import scatter` 现为五处未使用 import（可后续清理，清理后 B2 可摆脱 torch_scatter 运行时依赖）；`deploy/onnx_inference.py:75` docstring `idx_points_flat: flat indices for scatter merge` 措辞略陈旧。
- 生成物：`deploy/__pycache__/*.pyc`（`.gitignore:13` 已忽略，未污染仓库）。

## B3 (checkbox 18) — Step 6 改为 CPU 逆排列重排（方案 B 选 (b)）— 2026-09-16，GPU 5，无 git 操作

### 改动文件
- `deploy/CPP_trt4/src/pipeline.cpp`，两处区域（process_pointcloud / process_file），无其他文件改动。

### before / after（行号）
- process_pointcloud（before :211-246 → after :209-227）：
  - 删除：`// ---- Step 5: GPU scatter_mean 合并 ----` 整块（`d_out`/`d_cnt` 分配、`d_idx.upload()`、`launch_scatter_mean_kernel` 调用）、「原此处冗余 synchronize 已删」注释、原 Step 6 D2H（直写 result.logits + d_out 源）。
  - 保留/新增：`const int N_orig = num_points;`（原 :212，块内移到块首）；`result.logits.resize(N_orig*2)`（原 :239 逐字保留）；新 `std::vector<float> shuffled(N_orig*2)` 临时缓冲承接 d_src 乱序 logits；`CUDA_CHECK_THROW(cudaMemcpyAsync(shuffled.data(), d_src.data(), N_orig*2*sizeof(float), D2H, stream_.native()))`；`stream_.synchronize()`（取代原 sync，非叠加）；双重循环 `result.logits[idx_staging_[j]*2+c] = shuffled[j*2+c]`。
- process_file（before :416-453 → after :395-416）：
  - 同上删除项；`nvtxRangePushA("TAIL")` 保留、仅覆盖 D2H+sync，`nvtxRangePop(); // TAIL` 紧跟 sync 之后；**重排循环放在 TAIL pop 之后**（新 Step 6.5 注释，`// ---- Step 6.5: CPU 逆排列重排 (置于 TAIL pop 之后, 保持 TAIL 计时口径) ----`），ARGMAX_ACC push 之前。
- Step 4 前置块（两函数，原 :124-131 / :322-329）：删 `CudaBuffer d_idx(...)` 分配与 d_idx 注释行；首行注释「统一 scatter」→「CPU 逆排列重排」。
- 文件头流程注释 :15-16：`5. launch_scatter_mean_kernel → GPU 合并...` → `5. D2H 下载乱序 logits → CPU 按 idx_staging_ 逆排列重排 (B3: 去 scatter_mean)`；`6. 下载 merged logits, argmax` → `6. argmax → predictions`。
- 循环内 staging 注释（两函数）「循环后统一上传」→「循环后用于逆排列重排」。

### 取证链（编辑前逐行核实）
1. **d_src 布局**：`trim_transpose.cu:29-32` 每线程写 `d_src[(offset+j)*2+{0,1}]` ← TRT 输出 `d_out[j]`/`d_out[N_padded+j]`（channel-major→row-major 转置）→ d_src 第 `offset+j` 行 = 当前 chunk 第 j 点、通道 c。
2. **idx_staging_ 对应关系**：`pipeline.cpp:200-201`（/`:404-405`）`idx_staging_[offset+i] = idx_part[chunk_start+i]`，且 `offset`/`chunk_start` 每 chunk 同步累加 chunk_N → d_src 第 j 行 = 原始点 `idx_staging_[j]`。✅
3. **total_src ≡ N_orig**：C1/C2 单子云构造 `idx_points={shuffle(iota(0..N-1))}`，`total_src = Σ part.size() = N = num_points = N_orig`；chunking 路径 `split_oversized` 的 chunk_sizes 和 ≡ N，offset 逐 chunk 恰好耗尽 → D2H 尺寸 `N_orig*2` 与 d_src 分配 `total_src*2` 相等，无越界。✅
4. **覆盖/唯一性**：Fisher-Yates shuffle 输出是 0..N-1 的置换；idx_staging_ = idx_part 各 chunk 连续切片的拼接 → 每原始索引恰好出现一次 → resize(N_orig*2) 后每行恰好写一次，无漏写/无重叠写。✅
5. **本地变量名核实**：两函数均有 `N_orig`（:209 / :395）、`d_src`（:127 / :311）、`idx_staging_`（成员，pipeline.h:110）、`result` → spec 代码块零改名可移植。

### 编译
- 增量 `make -j`（build/ 复用 C7 的 `-DCUDNN_LIB=...` 缓存）→ **exit 0**；仅重编 `pipeline.cpp.o` + relink。
- 新二进制 `build/hpenet_trt_infer`：sha256 `49bce2900bd87430bdab86fa4ddd8fe94e0a9621cc70893b95354d57ec945504`，**1,596,440 B**（与 C7 二进制同为 1,596,440 B——删掉的内核调用/缓冲不改变段布局）。
- clangd 唯一告警：`pipeline.cpp:29 #include "scatter_mean.h" is not used directly` —— 属预期死代码状态（spec 要求保留）。

### 运行时 sanity（GPU 5，与 C7 同引擎/stats/data_dir/num_files=10）
- 逐文件 acc：0.9282/0.9297/0.9358/0.9288/0.9477/0.9347/0.9456/0.9446/0.9318/0.9450，**Mean 0.9372** —— 与 C7 记录**逐文件逐位一致**。
- 意义：count==1 下 CPU 逆排列与旧 GPU scatter_mean（out[idx]=logits）数值逐位等价，方案 (b) 兑现「只去 mean 归约开销、不改数值」的设计意图；C7 的 0.9372 口径不变，无精度回归。延迟终判仍属 F4（未跑 nsys）。

### 死代码状态报告（保留，未删）
- `src/scatter_mean.cu` / `include/scatter_mean.h` / `pipeline.cpp:29 #include "scatter_mean.h"` / `CMakeLists.txt:64 src/scatter_mean.cu`：`launch_scatter_mean_kernel` 已无调用者 → **全部死代码**，按 spec 保留（scatter_mean.cu 同时是兄弟工程 CPP_trt 的被测对象）。
- 已存在但未动的两处陈旧注释（不同文件，任务书禁改，仅报告）：`trim_transpose.cu:14`「scatter_mean 的 launch 才加」——scatter launch 已不存在；`pipeline.h:106-109` idx_staging_ 注释「pageable H2D 的驱动 staging / 旧上传已完成」——idx_staging_ 已无 H2D 上传。

### 残留扫描
- `grep -nE '\bd_out\b|\bd_cnt\b|\bd_idx\b|launch_scatter_mean' src/pipeline.cpp` → **0 命中（exit 1）**；`d_output`/`d_output_`（TRT 输出缓冲，与被删缓冲无关）仍在役。
- 未运行任何 git 命令；未改 .omo/ 任何文件；未跑 nsys / Python 管线。

## B4 验收（checkbox 19）— A+B+C 全栈 acc 复验 — 2026-09-16，GPU 3/4/5，无 git 操作，零文件改动

### 命令与耗时（均附 exit code）
- **Part1 ×2**（逐字符复用 P1/A4 命令，GPU 3，`/usr/bin/time -v` 计时）：
  `source /home/wangpeng/miniforge3/etc/profile.d/conda.sh && conda activate hpenet && CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml wandb.use_wandb=False mode=test --pretrained_path log/radar/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV/checkpoint/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV_ckpt_best.pth`
  → run1 **38.47s** exit 0；run2 **41.45s** exit 0。
- **Part2 ×2**：`CUDA_VISIBLE_DEVICES=3 python /tmp/opencode/ti10_baseline.py`（P1 原脚本未动）→ run1 **7.77s**、run2 **7.52s**，均 exit 0。
- **Part3 trt**：`CUDA_VISIBLE_DEVICES=4 python deploy/trt_inference.py --num_files 10 --compare` → **6.11s** exit 0。
- **Part3 onnx（plugin 默认模型）**：`CUDA_VISIBLE_DEVICES=5 python deploy/onnx_inference.py --num_files 10 --checkpoint <pinned>` → **3.78s exit 1**（issues #9）。
- **Part3 onnx（标准算子老模型，仅管线跑通证明）**：`... --onnx deploy/onnx_model_feat5_bn_sim.onnx --checkpoint <pinned>` → **15.24s exit 0**。
- **Part4 CLI**（GPU 5，`LD_LIBRARY_PATH` 含 pip cudnn）：`./deploy/CPP_trt4/build/hpenet_trt_infer --engine deploy/hpenet_v2_fp32.engine --stats deploy/CPP_trt/stats_feat5.json --data_dir data/RadarClassi/radarfullwl/raw --num_files 10` → **0.74s exit 0**。（第一次误从 CPP_trt4 工作目录跑、相对路径导致 exit 134 "Failed to open engine file"，属本次路径口径失误，非二进制问题。）

### Part1 三列指标（58 文件全量 test 集，2dp 口径）
| metric | pre-A 基线 | post-A | post-B run1 / run2 | Δ vs pre-A |
|---|---|---|---|---|
| OA | 92.05 | 91.98 | 91.98 / 91.98 | **−0.07** |
| mACC | 88.50 | 88.58 | 88.58 / 88.58 | +0.08 |
| mIoU | 77.02 | 76.92 | 76.92 / 76.92 | −0.10 |
| mP | 84.55 | 84.39 | 84.39 / 84.39 | −0.16 |
| mR | 88.50 | 88.58 | 88.58 / 88.58 | +0.08 |
| iou_valid | 90.79 | 90.71 | 90.71 / 90.71 | −0.08 |
| iou_invalid | 63.26 | 63.13 | 63.14 / 63.14 | −0.12 |
| prec_valid | 96.60 | 96.65 | 96.65 / 96.65 | +0.05 |
| prec_invalid | 72.51 | 72.12 | 72.13 / 72.12 | −0.38 |
| rec_valid | 93.79 | 93.65 | 93.65 / 93.65 | −0.14 |
| rec_invalid | 83.21 | 83.52 | 83.52 / 83.52 | +0.31 |

- **判定 PASS**：ΔOA = **−0.07pp < 0.3pp**；post-B 与 post-A 在 2dp 口径完全一致（仅 prec_invalid/iou_invalid 末位边界抖动 72.12↔72.13/63.13↔63.14，同 P1 记录的 2dp 确定性水平）→ **B1 数值惰性获系统级证实，无红旗**。
- 框架行为：pinned run 目录新增 2 个时间戳 log + `_test.csv` 追加 2 行、cfg.yaml/hpenet-ll.yaml 被重写（预期）；ckpt/训练 CSV/训练 log 未动。

### Part2 ti10
- 新值 **93.200669**（两次逐位一致）；记录参考 93.2147 / 93.2171 → **Δ = −0.014 / −0.016pp**，落在 A 波 shuffle 抽样变化（issues #6）预期内。
- 口径说明：ti10_baseline.py 的合并行仍是 scatter-mean 形式（脚本自身复刻代码、非 main.py 内路径）；B1 已证 bit 等价，B1 真实路径由 Part1 的 mode=test 覆盖。

### Part3 deploy 脚本
- trt_inference.py `_trt` 路径（FPSPrune 引擎变体）：逐文件 0.9282/0.9297/0.9358/0.9288/0.9477/0.9347/0.9456/0.9446/0.9318/0.9450，**Mean 0.9372** —— 与 C7/B3 记录及该引擎历史 0.9372 **逐位一致**。
- trt_inference.py `_pytorch` 路径（无 prune 模型，`--compare` 启用）：逐文件 0.9469/0.9104/0.9390/0.9265/0.9334/0.9187/0.9281/0.9296/0.9241/0.9420，mean 0.9299（seed-100 shuffle 口径，与 Python 参考 seed-0 口径不同，属正常变体差异，非退化）。
- trt_inference.py `_onnx` 路径：CLI 中被注释、不可选（`--onnx` 实参不生效）。
- onnx_inference.py：plugin 默认模型无法加载（issues #9）；标准算子老模型跑通 **exit 0 且无 TypeError**（A3③ 运行时证明）；其 ONNX acc 0.62~0.68 / PT acc 0.73~0.78 **无验收意义**（Aug-13 老模型权重 + patched PT 在 CPU 上 FPS 回退 zeros），仅作管线完整性证明。
- 回绑 bug 类复查（🔴）：trt 三站点与 onnx 站点均 `merged = torch.empty_like(all_logits_cat); merged[idx_flat] = all_logits_cat; return merged` —— 下游消费的是重排后 buffer；main.py :705-707 `all_logits = out` 回绑 + :709 消费。**无「建 out 不回绑」缺陷。**

### Part4 CLI
- Mean accuracy **0.9372**，逐文件与 B3 sanity 记录逐位一致。独立复跑成立。

### 完整性核验（运行后）
- 抽查点全部在位：`default.yaml:7 voxel_size: null`、`main.py:705-707`（B1）、trt_inference B2 三站点（:88-89/:114-115/:143-144）、onnx_inference B2（:120-121）与 A3③（:269 `.get('voxel_size', None)`）、A1（s3disRadar.py:73）、A3①（main.py:139-142）、A3②（common.py:69-73）。
- `find -newermt "2026-09-16 14:30"`（排除 log/build/data/.git/.omo/__pycache__）仓库内**零命中** → 本次运行未改动任何源文件；QA 产物全部落在 /tmp/opencode（b4_* 新文件）。
- 未运行任何 git 命令；GPUs 3/4/5 运行后还原空闲。

## F2 验收 — val/test 行独立复测（2026-09-16，GPU 3 / L20，无 git 操作，零文件改动）

### 命令（逐字符复用 P1/A4/B4 口径；全部 exit 0）
- **Part1 ×2（58 文件全量 test 集）**：
  `source /home/wangpeng/miniforge3/etc/profile.d/conda.sh && conda activate hpenet && CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml wandb.use_wandb=False mode=test --pretrained_path log/radar/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV/checkpoint/radar-train-hpenet-ll-ngpus1-20260907-170521-N3jVvf3C6aJCqvS9rLQ6vV_ckpt_best.pth`
  → run1 31.72s exit 0；run2 32.48s exit 0。
- **Part2 ×2（ti10 子集）**：`CUDA_VISIBLE_DEVICES=3 python /tmp/opencode/ti10_baseline.py`（P1 原脚本未动）→ run1 7.08s、run2 7.08s，均 exit 0。
- 生效 seed：`main.py:608 set_random_seed(0)`（grep 复核在位）；ti10_baseline.py:52 同。钉死 ckpt sha256 `130340c9...e42f1ec` 运行前/后均复核一致；加载 best_epoch=85。

### Part1 结果（58 文件全量，2dp 口径）与 Δ
| metric | pre-A 基线 | F2 run1 | F2 run2 | Δ(run1) | Δ(run2) |
|---|---|---|---|---|---|
| OA | 92.05 | 91.98 | 91.98 | **−0.07** | **−0.07** |
| mACC | 88.50 | 88.58 | 88.58 | +0.08 | +0.08 |
| mIoU | 77.02 | 76.92 | 76.92 | −0.10 | −0.10 |
| mP | 84.55 | 84.39 | 84.39 | −0.16 | −0.16 |
| mR | 88.50 | 88.58 | 88.58 | +0.08 | +0.08 |
| iou_valid | 90.79 | 90.71 | 90.71 | −0.08 | −0.08 |
| iou_invalid | 63.26 | 63.13 | 63.14 | −0.13 | −0.12 |
| prec_valid | 96.60 | 96.65 | 96.65 | +0.05 | +0.05 |
| prec_invalid | 72.51 | 72.12 | 72.13 | −0.39 | −0.38 |
| rec_valid | 93.79 | 93.65 | 93.65 | −0.14 | −0.14 |
| rec_invalid | 83.21 | 83.51 | 83.52 | +0.30 | +0.31 |

- 与 A4/B4 记录的 post-change 值（91.98/88.58/76.92/84.39/88.58）**逐位一致**（非转述，独立复跑测得）；prec_invalid/rec_invalid 的 ±0.3~0.4pp 少数类边界翻转再次可复现，谐波 iou_invalid 仅 −0.12~−0.13pp，与 spec §3.4 统计等价预期一致。

### Part2 ti10 结果
- F2 run1 mean OA = **93.200669**；run2 = **93.198293**（9/10 文件逐位一致，仅 0000069 抖动 92.444763↔92.420998，即 P1 记录的同一抖动文件）。
- 记录参考 93.2147 / 93.2171 → Δ(run1) = **−0.0140 / −0.0164pp**；Δ(run2) = **−0.0164 / −0.0188pp**。
- pooled（run1/run2）：OA 93.1738/93.1714，mIoU 80.2354/80.2302，mACC 91.0527/91.0513。

### 0.3pp 判据算术
- 全量集：|ΔOA| = |91.98 − 92.05| = **0.07pp < 0.3pp** ✅
- ti10：|Δmean OA| max = |93.198293 − 93.2171| = **0.019pp < 0.3pp** ✅
- 全部聚合指标（OA/mACC/mIoU/mP/mR/iou×2）|Δ| ≤ 0.16pp ✅

### 运行间噪声
- 全量集两次：OA/mACC/mIoU/mP/mR 2dp 逐位一致；iou_invalid 63.13↔63.14、prec_invalid 72.12↔72.13、rec_invalid 83.51↔83.52（≤0.01pp）→ 与 P1 记录噪声水平相同。
- ti10 两次：mean OA 差 0.0024pp（0000069 单文件抖动主导）→ 噪声 ≤0.01pp 口径成立。

### 完整性核验（运行后）
- 钉死 run 目录：ckpt（sha256 130340c9...）/训练 CSV/训练 log/events 四个文件 sha256 **逐字节未变**；`_test.csv` 12→16 行（每次 run 追加 header+row，框架行为）；新增 2 个时间戳 test log；cfg.yaml/hpenet-ll.yaml 被框架重写（cfg.yaml:21 仍 `voxel_size: null`）。**未删未移任何文件。**
- GPUs 3/4/5 运行后全部还原 3 MiB 空闲；未触碰 GPU 0/1/2/6/7；未运行任何 git 命令；仓库源文件零改动（仅按任务书 append 本 notepad）。
- QA 产物：/tmp/opencode/f2_part1_run{1,2}.{log,time}、f2_ti10_run{1,2}.{log,time}、f2_pinned_dir_{pre,post}.{ls,sha}。

### F2 VERDICT: APPROVE

## F1 终验收：spec §5.2 train 行（2026-09-16，GPU 3 / L20，无 git 操作，源文件零改动）

### 命令（逐字符；seed=47 依据：cfg.seed=null 运行期随机，pinned run 日志头部 `seed: 47`，必须显式钉回）
```
source /home/wangpeng/miniforge3/etc/profile.d/conda.sh && conda activate hpenet && CUDA_VISIBLE_DEVICES=3 python examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml wandb.use_wandb=False seed=47 epochs=1
```
- 新 run 目录：`log/radar/radar-train-hpenet-ll-ngpus1-20260916-144121-RbyvPCoLBL4qSaBE4RhBuU`（ckpt_best.pth 与 ckpt_latest.pth 均写入，events/cfg.yaml/hpenet-ll.yaml/CSV/log 齐备）。
- 耗时：14:41:21 → 14:43:39 ≈ **138s**（train epoch 进度条 01:23/351 batch + val + 框架自带的 58 文件 test ≈37s）。
- 日志头部核验：`seed: 47`、`opts: seed=47-epochs=1` 生效；`cfg.yaml` 保留 `voxel_size: null`（default.yaml:7 未动）。
- GPU 3 运行后还原 3 MiB 空闲；未触碰 GPU 0/1/2/6/7。

### epoch-1 控制台行（逐字，新 run log L896；基线为 pinned run L896，非重跑 0.0001 arm）
- 基线：`Epoch 1 LR 0.000001 train_miou 35.46, val_miou 39.45, best val miou 39.45, val_mp 56.08, val_mr 60.44`
- 新 run：`Epoch 1 LR 0.000001 train_miou 35.49, val_miou 38.53, best val miou 38.53, val_mp 55.80, val_mr 60.10`

### TB scalars @E1（events.out.tfevents 精确值，非控制台 2dp）
| metric | 基线 0.0001 | 新 null | Δ |
|---|---|---|---|
| train_loss | 0.712205 | 0.7116 | −0.0006 |
| train_miou | 35.46479 | 35.485279 | **+0.0205** |
| train_macc | 56.608124 | 56.644905 | **+0.0368** |
| train_mp | 53.710213 | 53.73027 | +0.0201 |
| train_mr | 56.608124 | 56.644905 | +0.0368 |
| val_miou | 39.451622 | 38.531616 | −0.920 |
| val_macc | 60.441769 | 60.097786 | −0.344 |
| val_oa | 62.463146 | 61.118965 | −1.344 |

- **train 判据（§5.2 train 行核心）**：所有 train 指标 |Δ| ≤ 0.037pp << 1.0pp ✅；train_loss Δ −0.0006。
- **val@E1 偏低属 warmup 首步（LR 1e-6）+ RNG 流偏移的正常噪声，不是验收判据**；与 A4 记录的 val 偏移（−0.954pp）同量级、方向一致，且收敛后 test acc 判据（F2 全量 −0.07pp）已另行走过。
- train 路径在本次计划中零改动（6 处 scatter/mean 编辑全在推理/测试侧）→ PASS 符合预期，且已实测确认而非假设。

### loss 发散 / NaN 检查（方法：日志全量大小写不敏感扫描 + 逐 batch 轨迹）
- `grep -icE "nan"` 日志 = **0**；`[Nn][Aa][Nn]` 变体 = 0。
- `grep -inE "inf"` = 146 命中，**全部为 `RadarClassi INFO:` 前缀的子串**（假阳性）；区分方法：`grep -inE "inf" | grep -viE "INFO"` → **0 命中**；词边界 `\bnan\b|\binf\b|\binfinity\b` → 0 命中。
- 逐 batch loss 轨迹（tqdm 进度条，523 个采样点，tee 捕获 `\r` 拆分后唯一值排序）：0.670 → 0.757 **平滑单调、无跳变**；loss 字符内 nan/inf = 0。TB train_loss@E1 = 0.7116（与基线 0.7122 同水平）→ **无发散、无 NaN/Inf** ✅。

### 完整性
- 运行前 pinned run 目录未动；运行后新目录保留（未删未移）。未修改任何源/配置文件；仅按任务书 append 本 notepad。未运行任何 git 命令。
- QA 产物：/tmp/opencode/f1_train_run.log（tee 捕获）、/tmp/opencode/f1_read_events.py（TB 读取脚本）。

### F1 VERDICT: APPROVE

## F3 验收：部署 Python 行（2026-09-16，GPU 4 / L20，无 git 操作，零文件改动，独立复核）

### 环境
- conda hpenet；python 3.10.20；torch 2.2.2+cu118；**onnxruntime 1.23.2**；**torch_scatter 2.1.2+pt22cu118**。
- 四个改动模块 import 干净：deploy/common.py、deploy/trt_inference.py、deploy/onnx_inference.py、deploy/onnx_backend.py。
- ⚠️ 过程纪律偏差（如实记录）：第一次 trt run 忘记 `CUDA_VISIBLE_DEVICES`，误用 **GPU 0** 跑 5.75s（进程退出后已释放，未干扰他人任务）；随后全部在 **GPU 4** 重跑，逐文件 acc 与误跑完全一致，结论不受影响。GPU 3 当时被并发任务占用（40% util），未触碰。GPUs 0-5 结束时全部空闲。

### 命令与结果（均附 exit code + 墙钟）
1. 默认调用 `CUDA_VISIBLE_DEVICES=4 python deploy/trt_inference.py --num_files 10`（无 --compare）→ **exit 0，5.82s**。实际只走 `_trt`（`_pytorch` 由 --compare 门控；`_onnx` 在 :213-216 / :287-292 被硬注释，CLI 不可选）。
2. `CUDA_VISIBLE_DEVICES=4 python deploy/trt_inference.py --num_files 10 --compare` → **exit 0，4.63s**。走 `_trt` + `_pytorch`。
3. `CUDA_VISIBLE_DEVICES=4 python deploy/onnx_inference.py --num_files 10 --checkpoint <钉死ckpt>`（默认 plugin onnx）→ **exit 1，3.55s**（traceback 见下）。
4. `CUDA_VISIBLE_DEVICES=4 python deploy/onnx_inference.py --num_files 10 --onnx deploy/onnx_model_feat5_bn_sim.onnx --checkpoint <钉死ckpt>` → **exit 0，12.96s**，10 文件全跑完、无 TypeError。

### 各路径 acc 与所用参考（B2 静默低 acc 对拍，逐条注明参考）
- `_trt`（FPSPrune 引擎变体；参考 = 该引擎在 ti10 的留档，任务书提供 + C7/B3/B4 记录）：
  逐文件 0.9282/0.9297/0.9358/0.9288/0.9477/0.9347/0.9456/0.9446/0.9318/0.9450，**Mean 0.9372**
  vs 留档 0.9372（逐文件逐位一致）→ **Δ = 0.0000pp** ✅
- `_pytorch`（无 prune 模型，可与 Python 参考比）：
  逐文件 0.9469/0.9104/0.9390/0.9265/0.9334/0.9187/0.9281/0.9296/0.9241/0.9420，**mean 0.9299**
  vs B4 同路径留档 0.9299（逐文件逐位一致）→ Δ = 0.0000；
  vs Python 参考（ti10 10 文件 mean OA：B4 最新 93.2007，P1 记 93.2147/93.2171）→ **Δ = −0.21pp**（< 0.3pp）✅
  ⚠️ 口径：`_pytorch` 用 preprocess_test 的 seed-100 shuffle，Python 参考用 seed-0 → −0.21pp 含 shuffle 变体成分（B4 已记，非退化）。
- `_onnx`：CLI 不可达（硬注释）。用 mock session 独立验证重排保留（见下）；acc 无验收意义（无可用 ONNX 模型，见 A3③ 节）。
- `--compare` 的 PredMatch 列 0.9323~0.9589（TRT 引擎变体 vs PT 无 prune 模型，两套独立合并实现互相对照）——若任一侧错序 pred_match 会 ~0.5，属系统级旁证。

### 回绑 bug 类系统级复验（重排确实被保留）
- 五站点代码逐行核验：均为 `merged = torch.empty_like(all_logits_cat); merged[idx_flat] = all_logits_cat; return merged` —— 返回即重排后 buffer，下游消费无「建 out 不回绑」缺陷；main.py:705-707 `out[idx_points] = all_logits; all_logits = out` 回绑在位、:711 消费 `all_logits.argmax`。
- /tmp/opencode/f3_reorder_check.py（mock 后端把子云位置编码进 logits，CPU 运行）：五个 helper 输出均满足 merged[i]==perm^{-1}[i]（原始点序，非子云序）、无 NaN、argmax 全 0；真实 0000068.ply（voxel_size=None）idx_flat 为 0..3706 全排列。
- 系统级判据：若合并被跳过/错序，逐文件 acc 会跌到随机置换水平（~0.5），不可能与留档逐位一致。

### A3③ 诊断确认（确认先前结论：先于改动的既有缺口，非本方案引入）
- 失败点行序：plugin onnx 在 **onnx_inference.py:201 `ort.InferenceSession(...)`** 即 exit 1：`Fatal error: hpenet:FPSPrune(-1) is not a registered function/op`；A3③ 在 **:269**，B2 站点调用在推理循环（:273/:286 之后）→ 失败先于全部方案改动行。
- 全仓 grep：**不存在任何 onnxruntime 侧自定义算子注册**（唯一命中是 CPP_onnx/include 的 ORT C++ SDK 头文件本身；onnx_backend.py 是 torch 导出侧 symbolic/traceable 实现）。plugin onnx 含 17 个 `hpenet` 域节点 + opset import `('hpenet',1)`；sim 老模型 691 节点全默认域。
- A3③ 运行时证明（本环境最强可用手段）：sim 模型 run 中 :269 被执行 10 次，且 EasyConfig 合并实测 `cfg.dataset.common.voxel_size` 为 None（NoneType）；旧代码 `float(None)` 必然 TypeError，本次 exit 0 无 TypeError → A3③ 修复真实生效。

### torch_scatter 残留（nit，任务书预告，确认仍在）
- `from torch_scatter import scatter` 仍在 trt_inference :87/:113/:142 与 onnx_inference :119/:162，位于函数体内、每次调用都执行 import → **仍构成运行时依赖**（缺包则 ImportError），但 `scatter` 已无任何使用点。清理可彻底解除依赖，属 F5 决定。

### F3 判定
- B2 五站点全部验证：4 站点真实推理 acc 与留档逐位一致（trt `_trt` Δ=0.0000 / trt `_pytorch` Δ=0.0000 同路径 + −0.21pp vs Python 参考）+ 1 站点（trt `_onnx`）harness 证明重排保留；onnx_inference 两站点随 sim run 真实执行 10 文件无异常。
- A3③：运行时无 TypeError，修复真实生效。
- 唯一无法执行的 onnx_inference ONNX acc 对拍系先于方案存在的 ORT 注册缺口（issues #9），与本方案 A/B/C 改动无关。

## 【更正】C7 节「端到端 ≥5% 口径达成」为过度声明 —— 2026-09-16（独立复核后追加，非改写历史）

> 本节更正上文中 **C7 节 C 段（约 L345）** 的原话：「spec CLI 行的『端到端 ≥5%』口径达成（−9.4%，但受 PLY_LOAD IO 噪声影响大），『部署口径 ≥5%』本单次 −1.6% 未达」。
> **「端到端 ≥5% 口径达成」这一表述不成立，收回。** 以下为独立重算（查询 `cpp_trt4_baseline.sqlite` 与 `cpp_trt4_after_voxel_removal.sqlite` 两份 nsys sqlite）的数字，覆盖原记录：

### 独立重算（单次运行，ms/文件）

| segment | baseline | after | Δ |
|---|---|---|---|
| PLY_LOAD | 6.208 | 5.299 | −0.910 |
| COORD_SHIFT | 0.007 | 0.006 | −0.001 |
| SUBCLOUD_LOOP | 2.460 | 3.402 | **+0.943** |
| TAIL | 1.312 | 0.762 | −0.549 |
| ARGMAX_ACC | 0.007 | 0.007 | −0.000 |
| VOXELIZE | 0.462 (n=10) | 0 (n=0) | −0.462 |
| **部署口径（去 PLY_LOAD）** | **4.247** | **4.177** | **−0.069 → −1.6%** |
| 端到端 | 10.455 | 9.476 | −9.4%（其中 93% 来自 PLY_LOAD） |

### 结论（逐条，供后续引用）
- **≥5% 判据在部署口径下 NOT established**：部署口径仅 −0.069 ms（−1.6%），远未达 5%。原「端到端 ≥5% 口径达成」的说法属口径混淆，已收回。
- **−9.4% 的端到端数字由 PLY_LOAD 主导**：PLY_LOAD 单独贡献 −0.910 ms（占端到端改善约 93%），属 IO/页缓存噪声，**不能作为去体素化收益**。
- **VOXELIZE 段确已消失**：10 → 0 事件；**独立查询两份 nsys sqlite 均确认**，非转述。
- **SUBCLOUD_LOOP 反而上升 +0.943 ms，大于被移除的工作（−0.462 ms）**：**属未解释异常**，已由专门 lane 进一步排查（不属本 notepad 任务范围）。
- **以上均为单次运行数字**，未做多次复跑；不应作为最终延迟判据，延迟终判仍归 F4。

## 【证据卫生】钉死 run 目录与 torch_scatter 残留（2026-09-16 追加）

1. **钉死 run 目录的 `*_test.csv` 已混入 8 行、跨口径**：`mode=test` 把产物写回 pretrained 所在目录，导致该 CSV 现含 **8 行结果**，混合了 baseline `voxel_size=0.0001` 的运行与改动后的运行。**后续读者必须按时间戳/行序分段，它不是单一口径记录。** 此外该目录的 `cfg.yaml` / `hpenet-ll.yaml` **已被框架重写**，不得作为原始训练配置信任；**checkpoint、训练 CSV、训练 log、TB events 四个文件保持完好，并已用 sha256 核验**（pinned run 训练侧证据链仍有效）。
2. **`torch_scatter` 残留 import（记录为 nit，不清理）**：`deploy/trt_inference.py`（3 处）与 `deploy/onnx_inference.py`（2 处）共 **5 个 `from torch_scatter import scatter`** 现已**未使用**。本 plan 只规定替换调用点，未要求清理 import，故仅记录为 nit；后果是这两个脚本**仍以 `torch_scatter` 为运行时依赖**（缺包即 ImportError）。是否清理属用户/评审范围外决定。

## F4 终验收：部署（CLI/C-API）+ 构建行 + 两项评审 blocker（2026-09-16，GPU 5 / L20，无 git 操作，零源文件改动，仅 clean rebuild 写入 build/）

### A. clean rebuild（评审 should-fix；resolve ✅）
- 命令（逐字符按任务书）：
  `rm -rf build && mkdir build && cd build && cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89" -DCMAKE_BUILD_TYPE=Release -DCUDNN_LIB=/home/wangpeng/miniforge3/envs/hpenet/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8` → **exit 0**；`make -j` → **exit 0**。
- 新二进制 `build/hpenet_trt_infer`：sha256 **`07c4c07595967e269e82523b42224689fd9f32c21cb27ddca1a940d7e27eb7a5`**，**1,596,440 B**（mtime 2026-09-16 17:49:54）。旧增量二进制 49bce290 已被覆盖（属预期）。link.txt grep voxelizer|fnv_hash = 0 命中。
- 两次注释改动（trim_transpose.cu:12-14、pipeline.h:106-109）已在树内；CLI 逐文件 acc 与 C7/B3/B4 记录**逐位一致**（0.9282/0.9297/0.9358/0.9288/0.9477/0.9347/0.9456/0.9446/0.9318/0.9450，Mean 0.9372）→ 注释级改动零行为影响 ✅。

### B. 重复延迟测量（评审 blocker 1 的测量部分；5 次新运行 + C7 单次，全 GPU 5）
- 方法与基线同：`nsys profile --trace=cuda,nvtx --force-overwrite=true -o /tmp/opencode/f4_run{r}` + `nsys export --type sqlite`（**注意：nsys 2022.4.2 的 export 默认把 .sqlite 写到 CWD，不是 .nsys-rep 旁边**——本轮首次 export 误落在仓库根，已删除、从 /tmp/opencode 重导出）。每轮 exit 0、墙钟 5.98~6.88s。
- 每轮前 nvidia-smi（GPU 5 均 3 MiB/0% 空闲；GPU 6/7 被他人 36 GiB 占用未触碰）；宿主 128 核，load avg 135~150（vLLM 引擎吃满 CPU）→ 宿主线程调度噪声显著。
- 独立重算（/tmp/opencode/f4_analyze.py，按文件分组聚合 sqlite，非转述 analyze_latency.py；与 analyze_latency.py 交叉核对一致），**部署口径 ms/帧**：

  | 运行 | 部署口径 | SUB | TAIL | VOXELIZE | PLY_LOAD | 端到端(分段和) |
  |---|---|---|---|---|---|---|
  | baseline (13:47, n=1) | **4.247** | 2.460 | 1.312 | 0.462 | 6.208 | 10.455 |
  | C7-after (14:14) | 4.177 | 3.402 | 0.762 | 0 | 5.299 | 9.476 |
  | F4 run1 | 3.875 | 2.559 | 1.306 | 0 | 5.262 | 9.138 |
  | F4 run2 | 3.492 | 1.735 | 1.748 | 0 | 4.958 | 8.449 |
  | F4 run3 | 3.624 | 2.004 | 1.611 | 0 | 5.013 | 8.637 |
  | F4 run4 | 4.099 | 3.087 | 0.996 | 0 | 13.696 | 17.795 |
  | F4 run5 | 3.617 | 1.943 | 1.664 | 0 | 4.887 | 8.505 |

- **部署口径判定算术（≥5% 判据）**：baseline 4.247（n=1，固定归档）。改动后 6 次运行 mean **3.814**、std 0.280、min 3.492、max 4.177 → **Δ = −0.433 ms = −10.2%（mean）**；仅 F4 5 次 mean 3.741 → **−11.9%**。6 次中 **5 次 ≥5%**（−7.7% ~ −17.8%），唯一未达的是评审引用的 C7 单次 −1.6%。**判定：≥5% 在均值与多数运行上成立；单次证据不足以成立（评审对该单次判据的否定正确），成立性靠重复测量**。−0.433ms 中 −0.462 直接来自 VOXELIZE 段消失（SUB+TAIL 六次均值 4.22 vs 基线 3.77 抵消部分），收益非噪声伪装。
- 端到端口径：除 run4（PLY_LOAD 被 IO 挤到 13.7ms）外 5 次 mean 8.84 → −15.4%，但 93%+ 变动来自 PLY_LOAD（IO/页缓存噪声），**不宣称其为收益**（评审已驳回该口径）。
- VOXELIZE 事件：6 份改动后 profile 全部 **0 事件**（基线 10）；GPU kernel 总 2.614~2.619 ms/文件（基线 2.686，−0.067，即删除的 hash+sort 内核）→ 删除确认 ✅。

### C. SUBCLOUD_LOOP 异常结论：**测量噪声 + 轻微归因漂移，非真实回归**
- 六次改动后 SUB 均值 **2.455 ≈ 基线 2.460** → C7 单次 +0.943 是离群样本（改动后 SUB 跨运行 1.735~3.402，std 0.63）。
- 噪声机理（证据）：C7 after profile 中宿主 cuLaunchKernel 单次调用阻塞达 **674µs**（正常 5-8µs），device 时间线出现 200-620µs GPU 空洞；同机 128 核 load 100-150（GPU 6/7 的 vLLM 挤占 CPU）→ NVTX 宿主侧区间被调度等待撑大。f4_run4 的 PLY_LOAD 13.7ms 是同类噪声的 IO 侧表现。
- 新 CPU 工作成本微基准（/tmp/opencode/f4_microbench，链接仓库 random_util.cpp）：shuffle 0.03~0.14ms、逆排列重排(2N 散布写) **0.002~0.03ms**、coord_shift 0.02~0.07ms（N=3707~7837）→ 无法解释 +0.94ms。
- 归因漂移（小）：新代码 Step 3（shuffle）与 Step 6.5（重排）**故意在全部 NVTX 区间之外**（B3 把重排放 TAIL pop 后）；WALL−Σ6段=UNCOVERED：基线 0.012 → 改动后 0.052~0.105 ms/文件 → 区间基本穷尽，隐藏量仅 ~0.05-0.1ms 的新 CPU 工作。

### D. C-API（clean rebuild 后，评审 blocker 3 部分可执行项）
- 驱动：/tmp/opencode/capi_driver.cpp 重编译 + 对新 clean 对象重链接（复用 link.txt、main.cpp.o 换驱动 .o）→ `/tmp/opencode/capi_driver_f4` 链接 exit 0；运行 10 文件 **exit 0**。
- 三入口逐文件 acc 全部 = CLI（0.9282/0.9297/0.9358/0.9288/0.9477/0.9347/0.9456/0.9446/0.9318/0.9450），**mean 0.9372**；四路预测源（CLI dump / process_file / process_inmemory / infer_and_update）**逐位一致**；GT 备份一致断言通过。
- **「Python 参考 <0.3pp」判据算术（两读法）**：C-API/CLI 93.72% vs Python 参考 ti10 93.2007（B4/F2）→ Δ = **+0.519pp**；vs 93.20（任务书口径）→ **+0.52pp**。**两读法均 > 0.3pp → 字面判据 FAIL**。
- **但属规格缺陷（非测量缺陷），需用户裁定，不静默通过**：C-API 只能跑 FPSPrune(keep_rate=0.75) TRT 引擎变体，Python 参考是无 prune 的 PyTorch 模型——**不同模型变体**。同变体对拍：C-API vs `trt_inference.py _trt`（同引擎）= **Δ0.0000pp**（F3 亦证）；无 prune 变体 vs Python 参考 = `_pytorch` −0.21pp < 0.3pp ✅；去体素化本身的 acc 判据由 F2（val/test ΔOA −0.07pp）覆盖。0.52pp 是**变体差**，非体素化差。
- CPP_trt3 前后对拍（真实 before）仍不可用（外部账户/额度限制），缺口记录在案，未尝试。

### E. Orin 可达性
- 唯一引用：`deploy/cmake_tensor.sh:36` 的 `adas@192.168.137.40`。本机在 192.168.1.0/24（ip route 无 137.0/24 路由），`getent hosts` 无条目，`ping -c1 -W2` **100% 丢包**。**不可达**。未做任何 SSH/SCP/凭据尝试。
- 附带：`deploy/measure_orin.sh` 系 Orin 上执行的脚本（BIN=./hpenet_trt_infer），本机无法代跑。

### QA 产物（全部 /tmp/opencode，仓库零新增）
f4_clean_cmake.log / f4_clean_make.log / f4_cli_clean_run.log / f4_cli_dump_run.log / f4_measure.sh / f4_run{1..5}.{nsys-rep,sqlite,_nvidia_pre.txt,_uptime_pre.txt,_nsys_stdout.log,_export.log} / f4_analyze.py / f4_microbench.{cpp,o 无} / f4_capi_driver.{cpp 复用 C7,o} / capi_driver_f4 / f4_capi_out/ / f4_capi_dump/ / f4_cli_dump/ / f4_capi_compare.py / f4_capi_driver.log。
- 过程中唯一事故：首次 nsys export 因 CWD 默认落盘把 5 个 f4_run*.sqlite 写进仓库根 → 已删除，树已核验干净（find -newermt 仅 build/）。未运行任何 git 命令。

## 评审 blocker 3 补齐：CPP_trt3 真·before 对拍（2026-09-16，GPU 5 / L20，无 git 操作，源文件零改动，append）

### 目标
填补 issues.md「评审驳回记录」blocker 3（C-API 去体素化前后从未对拍）。CPP_trt3 = 未改动的 blueprint，即 CPP_trt4 的忠实 before 镜像。**未改 CPP_trt3 任何文件**；build/ 为 gitignored 重建产物（clean rebuild 属任务指定动作）。

### 构建（clean rebuild，与 C7/F4 同法）
- `rm -rf build && mkdir build` → `cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89" -DCMAKE_BUILD_TYPE=Release`（无 -DCUDNN_LIB）→ **exit 0**；`make -j16` → **exit 2**（复现 cuDNN 链接问题：`undefined reference to cudnnGetErrorString@libcudnn.so.8` 等，来自 libnvinfer_plugin.so 的传递 NEEDED；28.82s 墙钟，全部 .o 已编译）。
- `cmake .. <同上> -DCUDNN_LIB=/home/wangpeng/miniforge3/envs/hpenet/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8` → **exit 0**；`make -j16` → **exit 0**（仅 relink，0.33s）。
- 产物：`deploy/CPP_trt3/build/hpenet_trt_infer` **3,777,896 B**，sha256 **`1377cf839ad050fc48d54ad9ecdb6a050c087a874b808c99e4b4effc1cb0b798`**；plugins 静态库 1,429,650 B。注：同尺寸下 sha 与 P2 基线（0b52d744）及上个 lane 遗留 build（9f13ce93）均不同 → 本机链接产物非逐字节可复现，**但 CLI 逐文件 acc 与 P2 基线逐位一致（见下），忠实性由预测一致性证明**，非由 sha 证明。
- 运行期仍需 `LD_LIBRARY_PATH` 含 pip cuDNN 目录（否则 --help 也 exit 127）；`--help` exit 0。

### CLI before（默认 voxel_size=0.0001f，main.cpp:31；help 文案写 0.02 系过时）
- 两次运行逐文件 acc **逐位一致且与 P2 基线逐位一致**：0.9396/0.9294/0.9456/0.9204/0.9460/0.9378/0.9449/0.9408/0.9142/0.9388，**Mean 0.9358**（subcloud 全 1）→ 本重建 = P2 基线二进制的行为等价物 ✅。

### C-API before（wrapper 硬编码 voxel 0.02f，trt_inference_wrapper.cpp:194；构造签名 pipeline.h:45-52 证实 0.02f 绑定 voxel_size，seed 默认 100）
- 驱动：复用 C7 的 /tmp/opencode/capi_driver.cpp 源码，对 CPP_trt3 对象重编译重链接（link.txt 中 main.cpp.o 换驱动 .o）→ `/tmp/opencode/capi_driver3`，链接 exit 0，3,786,000 B，sha256 3cca1c25…；运行 10 文件 **exit 0**（1.18s 墙钟）。
- 三入口（process_file / process_inmemory / infer_and_update）逐位一致；**GT 备份（.gt.valid.bin，调 infer_and_update 前）与原始 PLY label 全等** → 原地覆盖 valid 属实、备份法可对拍（spec §5.2 警示再次实证）。
- C-API 预测与「CLI --voxel_size=0.02」逐位一致 → 驱动接线正确（同一体素路径）。
- **C-API before 逐文件 acc = 0.9563/0.9266/0.9567/0.9404/0.9548/0.9445/0.9374/0.9535/0.9375/0.9548，Mean 0.946241 ≈ 0.9462**；voxel 0.02 下 10 文件均 **subcloud=2**（max_count=2，voxelizer.cu 按 Python `range(count.max())` 语义产子云）+ scatter_mean 投票。

### 对比表（评审要求的 before/after）
| path | before (CPP_trt3) | after (CPP_trt4) | Δ |
|---|---|---|---|
| CLI | **0.9358** | 0.9372（C7/B3/F4 记录） | **+0.0014（+0.14pp）** |
| C-API | **0.9462** | 0.9372（F4 记录，三入口逐位） | **−0.0090（−0.90pp）** |

- CLI 行：+0.14pp = C7 已查明的 shuffle 输入序差异（FNV 序 vs 扫描序 iota），在 §3.3/§3.4「非 bit 等价、按 acc 验收」框架内。
- **C-API 行是新事实：before 显著高于 after。** 0.02f 体素 + scatter_mean 投票在 C-API 路径上本是有精度增益的（10 文件 mean −0.90pp，逐文件 −2.81 ~ +0.82pp）；去体素化后 C-API 预测即 CLI 预测（单子云直通）。**「before 和 after 相同」的假设不成立；差异方向与 CLI 行相反。**

### 「<0.3pp vs Python 参考」判据裁定（附算术）
- Python 参考（ti10，无 prune 模型）：OA 93.20（B4/F2 记录 93.2147/93.2007 两口径）。
- 字面读法：after 93.72% − 93.20% = **+0.52pp > 0.3pp → FAIL**；且 **before 也 FAIL**：CLI before 93.58% → **+0.38pp**；C-API before 94.62% → **+1.42pp**。即：**未改动的 before 镜像本身也无法通过该判据** → 判据对 TRT/C-API 路径不可判别（pristine baseline 都过不了线），证实 issues.md #10「规格缺陷」判断，且本 lane 补上了 before 侧证据。
- 同变体读法（F4）：C-API vs `_trt`（同引擎）= 0.0000pp；`_pytorch` vs Python 参考 = −0.21pp；去体素化本身（F2）= −0.07pp —— 均 < 0.3pp ✅。
- **裁定：该行判据对 TRT/C-API 路径是规格缺陷（比较了 FPSPrune(0.75) 引擎与无 prune PyTorch 两个模型变体），需用户决定（改为同变体对拍，或接受变体基准）。不静默通过。** 另新增待用户决策项：C-API 路径去体素化带来 −0.90pp 的实测精度损失（体素投票增益被移除），这是判据当前完全未覆盖的、该路径自身的真实代价。

### 产物清单（全部 /tmp/opencode，未清理）
trt3_cmake_noflag.log / trt3_make_noflag.log（exit 2 证据）/ trt3_cmake_flag.log / trt3_make_flag.log / trt3_cli_run1.log / trt3_cli_run2.log / trt3_cli_run_v002.log / capi_driver3.{o,cpp 复用 C7 源码} / capi_driver3（3cca1c25…，3,786,000 B）/ trt3_capi_driver.log / capi3_compare.py / trt3_capi_compare.log（exit 0，三断言 PASS）/ capi3_cli_dump_default/（20）/ capi3_cli_dump_v002/（20）/ capi3_out/（10）/ capi3_dump/（30）。仓库零新增（build/ 为 gitignored 重建）；engine sha256 46c8cb47… 与 stats 全程未变；GPU 5 已还原（3 MiB/0%）。未运行任何 git 命令。

## F4b 终审复测 — 成对重复延迟测量：CPP_trt3(before) vs CPP_trt4(after)（2026-09-16，GPU 5 / L20，无 git 操作，源文件零改动，append）

### 背景
独立评审驳回 F4 的单次对比（C7 单次部署口径 −1.6% < 5%，SUBCLOUD_LOOP +0.943 未解释）。本 lane 用**成对交替重复测量**裁定：before = CPP_trt3（含体素化，含 NVTX VOXELIZE 段，CLI 默认 voxel_size=0.0001f），after = CPP_trt4（去体素化）。两二进制均未重建、未改动；顺序 **B,A,B,A,B,A,B,A**（交替摊薄热漂移/宿主负载漂移），每次全新进程，共 8 次 nsys 运行（n=4 每侧）。

### 运行前取证与占用
- 二进制 sha256（实测，与任务书一致）：B `1377cf839ad050fc48d54ad9ecdb6a050c087a874b808c99e4b4effc1cb0b798`（3,777,896 B）；A `07c4c07595967e269e82523b42224689fd9f32c21cb27ddca1a940d7e27eb7a5`（1,596,440 B）。engine sha `46c8cb47…2094867`（14,679,492 B）；stats `bbda0bf5…f1bee1`（296 B）；data 339 PLY；--num_files 10 → 0000068..0000077。
- 每次运行前 nvidia-smi：GPU 5 恒 3 MiB/0%（其余 0-4 同空闲；6/7 被他人 36 GiB 占用，未触碰）；load avg 94.9~117.0（宿主噪声与 issues #12 同源）。运行后 GPU 5 还原 3 MiB 空闲。
- 每次 nsys exit 0、export exit 0，墙钟 5.84~6.69 s/run。CLI acc：B 四次全为 **Mean 0.9358**（= 归档 CPP_trt3 基线，逐位一致）、A 四次全为 **Mean 0.9372**（= C7/B3/F4 留档）→ 两侧镜像忠实。

### 逐运行段表（ms/文件，10 文件 mean；独立重算 = 每文件按 PLY_LOAD 分组的段和均值，sqlite 直查 NVTX_EVENTS eventType=59；与 analyze_latency.py 交叉核对逐位一致）

| 运行 | PLY_LOAD | VOXELIZE | COORD_SHIFT | SUBCLOUD_LOOP | TAIL | ARGMAX_ACC | **部署口径** | 端到端(段和) |
|---|---|---|---|---|---|---|---|---|
| B1 | 14.184 | 0.613 | 0.009 | 3.292 | 0.836 | 0.008 | **4.758** | 18.942 |
| A2 | 10.898 | 0 | 0.008 | 2.825 | 1.110 | 0.005 | **3.947** | 14.846 |
| B3 | 4.916 | 0.483 | 0.006 | 2.005 | 1.732 | 0.011 | **4.236** | 9.152 |
| A4 | 4.841 | 0 | 0.006 | 1.824 | 1.743 | 0.004 | **3.576** | 8.417 |
| B5 | 13.366 | 0.697 | 0.009 | 3.195 | 1.030 | 0.009 | **4.939** | 18.305 |
| A6 | 6.823 | 0 | 0.007 | 2.719 | 1.141 | 0.005 | **3.871** | 10.694 |
| B7 | 4.931 | 0.408 | 0.006 | 1.814 | 1.779 | 0.007 | **4.014** | 8.945 |
| A8 | 6.290 | 0 | 0.006 | 2.136 | 1.557 | 0.004 | **3.703** | 9.992 |

### 分组统计（n=4 运行；样本 std）
| 段 | B mean±std (min..max) | A mean±std (min..max) | Δ(A−B) |
|---|---|---|---|
| PLY_LOAD | 9.349±4.435 (4.916..14.184) | 7.213±2.248 (4.841..10.898) | −2.136（页缓存噪声，不参与判据） |
| VOXELIZE | 0.550±0.112 (0.408..0.697) | **0（n=0 事件）** | **−0.550（删除确认）** |
| COORD_SHIFT | 0.007±0.001 | 0.006±0.001 | −0.001 |
| SUBCLOUD_LOOP | 2.577±0.671 (1.814..3.292) | 2.376±0.413 (1.824..2.825) | −0.201（A 反而更快，非回归） |
| TAIL | 1.344±0.417 (0.836..1.779) | 1.388±0.271 (1.110..1.743) | +0.043（噪声内） |
| ARGMAX_ACC | 0.009±0.001 | 0.004±0.001 | −0.004 |
| UNCOVERED | 0.015 | 0.073 | +0.058（新 CPU 工作出区间，见下） |
| **部署口径** | **4.487±0.434 (4.014..4.939)** | **3.774±0.167 (3.576..3.947)** | **−0.713 = −15.9%** |
| 端到端 | 13.836±4.793 | 10.987±2.375 | −20.6%（93% 由 PLY_LOAD 贡献，不作收益口径） |

### 判据算术（部署口径 ≥5%）
- mean Δ = 3.774 − 4.487 = **−0.713 ms → −15.9%** ✅
- 漂移控制相邻成对（B1/A2、B3/A4、B5/A6、B7/A8）：−17.0% / −15.6% / −21.6% / **−7.8%** → **四对全部 ≥5%** ✅
- 分布完全分离：max(A)=3.947 < min(B)=4.014；配对 t 检验 t=4.51 (df=3, p≈0.01)，未配对 t=3.07 (df=6, p≈0.011) ✅
- 收益构成：−0.550(VOX 删除) −0.201(SUB) +0.043(TAIL) −0.004(ARG) −0.001(COORD) = −0.713 ✓。GPU kernel 总 26.91→26.16 ms/run（−0.075 ms/文件）；B-only 内核 = fnv_hash_kernel + cub radix sort×3 + scatter_add/div_kernel，A-only = 空集。
- 与旧归档对照：本 lane 实测 B 部署口径 4.014~4.939（归档单次 4.247 落在区间内），A 3.576~3.947（F4 六次 3.492~4.177 同分布）→ 评审引用的 C7 单次 −1.6% 是「B 中低样本 vs A 高样本」的抽签结果，非真实水平。

### SUBCLOUD_LOOP 终裁：**测量噪声 + 轻微归因漂移，非真实回归**（证据链）
1. 成对重复：B SUB 2.577±0.671 vs A SUB 2.376±0.413 → A 数值上更低；Δ −0.201 仅为 B 自身 std 的 0.3 倍，统计不可区分。
2. 每文件跨 8 运行 SUB 极差 1.3~2.8 ms（如 f07: 1.531~4.326）；全部 80 个文件样本中最大值 4.562 属 **A8-f00（after 侧）** → C7 的 +0.943 只是该噪声包络的 ~1/3，纯属抽签。
3. 归因漂移（小、已量化）：B3 有意把新 CPU 工作放在区间外——A 的 Step 3 shuffle（pipeline.cpp:290-294）在 COORD_SHIFT pop 之后、SUBCLOUD_LOOP push 之前，Step 6.5 逆排列重排（:414-416）在 TAIL pop 之后、ARGMAX_ACC push 之前；区间外合计 UNCOVERED 0.015→0.073（+0.058 ms/文件）。微基准（f4_microbench 复跑 3 次，N=3707~7837）：shuffle 0.028~0.142 ms + 重排(2N 散布写) 0.002~0.014 ms → 与 +0.058 吻合。
4. 真实回归排除：A 无新增 GPU 内核；2N 散布写成本 0.002~0.014 ms；两镜像每 chunk 提交同一 stream FIFO 链，唯一 sync（TAIL D2H）未变 → 无丢失异步重叠。
5. 旁证：同运行内 SUB 与 TAIL 反相关（CPU enqueue 慢→sync 等待短），SUB+TAIL B mean 3.92 vs A 3.76 → 宿主调度在宿主侧两区间间再分配，不改总量。

### 记录项（E，不重测）
- C-API 前后对拍已由他 lane 解决：before（CPP_trt3 硬编码 0.02f）= **0.9462**，after（CPP_trt4）= **0.9372** → **−0.90pp**；0.02f 多子云投票（10 文件均 subcloud=2）是 2 路集成，CLI 路径从未有过。
- Orin（192.168.137.40）本机不可达（无路由、ping 100% 丢包）——环境缺口，未尝试任何访问。
- 「<0.3pp vs Python 参考」判据**非判别**：未改动的 before 镜像本身也 FAIL（CLI before 93.58% vs 参考 93.20 → +0.38pp；C-API before 94.62% → +1.42pp）→ 规格缺陷（FPSPrune 引擎变体 vs 无 prune PyTorch 模型），待用户裁定。

### QA 产物（/tmp/opencode/f4b_work/，全部保留）
f4b_measure.sh / f4b_analyze.py / f4b_run{1,3,5,7}_B.{nsys-rep,sqlite} / f4b_run{2,4,6,8}_A.{nsys-rep,sqlite} / 每运行 _{nvidia_pre,uptime_pre,bin_sha,nsys_stdout,export}.*。仓库零新增；未运行任何 git 命令；未触碰 GPU 0/1/2/6/7。

### F4 LATENCY VERDICT: **PASS**（部署口径 ≥5%：mean −15.9%，4/4 相邻成对 ≥5%，分布完全分离）

## B1 注释修正（2026-09-16，仅改 `examples/segmentation/main.py:703` 一行注释，无代码改动，无 git 操作）
- before `:703`：`            # average merge overlapped multi voxels logits to original point set`（B1 把 scatter('mean') 换成 `empty_like`+索引写后已失实：voxel_size=null 只有 1 个子云，每行恰写一次，是纯逆排列，非平均/合并）。
- after `:703`：`            # reorder sub-cloud logits back to original point order (no mean reduction)`（措辞对齐 `deploy/trt_inference.py` / `deploy/onnx_inference.py` 同操作注释）。
- 代码三行零改动证明：`:704-707` 提取切片 md5 前后均为 `73a4d061b386dd87382659f50b0587e4`，diff 空。`python3 -m py_compile examples/segmentation/main.py` exit 0。未碰 `:139-142`(A3①)、`deploy/*`、`.omo/plans/`；未跑训练/测试；未运行任何 git 命令。

## 防御性运行时不变量检查 total_src==N_orig（2026-09-16，GPU 5 / L20，无 git 操作，仅改 `deploy/CPP_trt4/src/pipeline.cpp`）

### NDEBUG / assert 取证（任务 §3）
- `deploy/CPP_trt4/CMakeLists.txt` **无**任何 `NDEBUG`/编译旗标行（107 行全文读完）；Release 的 `-DNDEBUG` 来自 CMake 内置 `CMAKE_CXX_FLAGS_RELEASE`。实测取证：`build/CMakeFiles/hpenet_trt_infer.dir/flags.make` 中 `CXX_FLAGS = -O3 -DNDEBUG -std=gnu++17`、`CUDA_FLAGS = -O3 -DNDEBUG ...` → **本工程 Release 构建确实定义 NDEBUG，裸 `assert()` 会被完全编译掉**，任务书 §3 的警告属实。
- 项目内 `assert(` 扫描（CPP_trt4 全树）：仅 8 命中 = `trt_engine.cpp:34 static_assert` + `test_rpc.c` 6 个 `_Static_assert`（编译期，不受 NDEBUG 影响）。**零运行时 `assert()` 先例** → 与「用 throw 而非 assert」决策一致。

### 异常类型选择：`std::runtime_error`（依据）
- `include/cuda_utils.h` 无异常宏；实际宏在 `include/logger.h:69-77`：`CUDA_CHECK_THROW` 失败时 `throw std::runtime_error(...)`（注释明言「用于业务路径——异常可被上层 try/catch 捕获」）。pipeline.cpp 本文件两处 `cudaMemcpyAsync` 即用 CUDA_CHECK_THROW → 该文件业务路径惯例 = runtime_error。
- `subcloud_utils.cpp` 的 4 处 `throw std::invalid_argument`（:22/:92/:159/:163）均为**调用方参数非法**的前置条件（如 `pad_subcloud` 的 N/min_n 校验）。本不变量 `total_src != N_orig` 不是调用方参数错误，而是**内部状态不一致（程序 bug）** → 语义上更贴近 runtime_error（与同文件 CUDA_CHECK_THROW 同型），故弃 invalid_argument。

### hunks（新行号，对称两处 + include）
- include 块：`#include <random>` 之后新增 `#include <stdexcept>`、`#include <string>`（§3.5 显式包含，不依赖传递包含）。
- process_pointcloud（:211 之后插入 :213-219）：
  ```
  211:    const int N_orig = num_points;
  212:
  213:    // 不变量: 去体素化后恒为单子云 (由构造保证), split_oversized 的 chunk 尺寸和 ≡ N → total_src 必等于 N_orig
  214:    if (total_src != N_orig) {
  215:        throw std::runtime_error(
  216:            "pipeline: total_src (" + std::to_string(total_src) +
  217:            ") != N_orig (" + std::to_string(N_orig) +
  218:            ") - single-subcloud invariant broken");
  219:    }
  ```
- process_file（:405 之后插入 :407-413，同文对称；NVTX 7/7 push/pop 平衡未动——检查位于 SUBCLOUD_LOOP pop 之后、TAIL push 之前，不涉及区间）。
- 其余逻辑零改动：子云循环 / d_src 分配 / D2H 拷贝 / 逆排列循环 / result.logits.resize 全部逐字未动（编辑前后 diff 仅上述 3 处 + include 2 行）。

### 构建（增量，任务书指定口径）
- 命令（逐字符）：`cd deploy/CPP_trt4/build && cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89" -DCMAKE_BUILD_TYPE=Release -DCUDNN_LIB=/home/wangpeng/miniforge3/envs/hpenet/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8 && make -j` → **exit 0**；仅重编 `pipeline.cpp.o` + relink（其余目标 Built target 未动）→ 增量行为正常。
- **新二进制 `build/hpenet_trt_infer`：sha256 `eeea6ff1eed12c5e182ef4daf299e658d4ac902c25077e0bc23a42285d0859d4`，1,596,440 B**（与 F4 基线同尺寸，+8 行代码落在对齐填充内）。
- **clean rebuild 不需要，理由**：(1) 本任务未增删源文件、未改 CMakeLists → C7 学到的「删文件后旧 link.txt 残留引用必须 clean」情形不适用；(2) 增量 link 正常（exit 0、无新警告，唯一告警仍是既有 `scatter_mean.h` unused include，spec 要求保留的死代码）；(3) F4b 已实证本机链接产物 sha 连 clean rebuild 也逐次不同，可信度由 CLI acc 逐位一致证明；(4) 任务书明示最终 clean rebuild 归后续 lane。
- **throw 确实存活于 NDEBUG 二进制（非 assert 的可编译性证明）**：`strings` 命中 `") - single-subcloud invariant broken"` + `nm -C` 见 `std::runtime_error` 构造引用；`pipeline.cpp.o` 反汇编 6 个 `__cxa_throw` 重定位 = 旧 2 空云检查(:88/:277) + 新 2 不变量检查(:215/:409) + 2 个 CUDA_CHECK_THROW 宏展开(cudaMemcpyAsync)，计数吻合。

### CLI 验证（GPU 5，任务书 §5 口径）
- `CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=<pip cudnn dir>:$LD_LIBRARY_PATH ./deploy/CPP_trt4/build/hpenet_trt_infer --engine deploy/hpenet_v2_fp32.engine --stats deploy/CPP_trt/stats_feat5.json --data_dir data/RadarClassi/radarfullwl/raw --num_files 10` → **exit 0**。
- 逐文件 acc：0.9282/0.9297/0.9358/0.9288/0.9477/0.9347/0.9456/0.9446/0.9318/0.9450，**Mean accuracy 0.9372** —— 与任务书 §7 基线（F4 二进制 07c4c075…）**逐文件逐位一致**；每文件 `subcloud=1`。
- 检查不可触发得证：10 文件 × 2 函数路径全走、无异常、acc 零漂移。GPU 5 运行前 util 12%（他人低负载常驻训练任务驻留 44GB，未触碰其进程），运行后本进程退出。
- 未跑 nsys（专用 lane 负责）；未跑 Python 管线；未触碰 GPU 0/1/2/6/7；未运行任何 git 命令；未改 `.omo/plans/`、CPP_trt3、scatter_mean.{cu,h} 及其 CMake 条目。
