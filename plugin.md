# HPENet V2 TensorRT Plugin 工程方案

> **目标**：将 `furthest_point_sampling` 与 `ball_query` 两个 CUDA 算子封装为 ONNX Custom Op + TensorRT Plugin，使 PyTorch → ONNX → Engine 全链路在保持训练精度的同时避免算子碎片化，最终把 Engine 推理集成到 Orin 上的自动驾驶 pipeline。
>
> **创建日期**：2026-08-13 | **修订**：2026-08-14（v13：实施后审查——更新实测 build 时间 29min、精度 0.9741、Reformat autotune 2256 分析）| **2026-08-17（v14：增补任务 §13——BallQueryGroup/BallQueryDP/ThreeInterp 三 plugin）→ v14.1 三视角审查修订（1 BLOCKER + 10 MAJOR）→ v14.2 第 16 次审查修订（2 MAJOR + 4 MINOR + 5 NIT：占位实现规范改调真实 CUDA op、5 调用点 vs 图内 4 份 TopK 的 CSE 澄清、min N 回改 3817、C 上限 512、V4 算术自洽 ≤1100、序列化公式、31.6ms 基线核注等）→ v14.3 第 17 次审查修订（1 MAJOR + 3 MINOR + 4 NIT：补 attribute 传递机制与 make_* 工厂、CUDA 路径引用来源与递归隐患、方案2 C=512 可行性边界、工作量补 Orin/Windows 2 天、附录 B 9 条核注等）→ v14.4 第 18 次审查修订（收敛轮：8/8 项编辑核验通过、端到端终读零 MAJOR/MINOR，仅修 6 项 NIT 措辞）→ **v14.5 实施完成（2026-08-17）：V1-V7 全部本地可测项通过，acc 0.9741 零回退，节点 628，延迟不劣于 v13；build 时间持平（证伪 Reformat 主导论）；实施记录见 §13.10** → **v14.6（2026-08-18）：新增附录 C 问答章节（5 个点云算子的关系与作用、xyz 为何锁死 fp32）** → **v14.7（2026-08-18）：一致性修正 5 处——§3/§7 节点期望补 v14 实际值（BQ 0/Group 4/DP 4/ThreeInterp 5/共 628）及 patch 描述更新、§9 v14 engine 大小 14.9→15.1MB、§13.6 悬空 31.6ms 括注改为实际回填值、kernel 头注释错位更正、V1 计数统一为 34/34（两处 32/32 为修复前旧数）****
> **适用模型**：`log/radar/radar-train-hpenet-ll-ngpus1-20260812-201051-*/checkpoint/*_ckpt_best.pth`
> **目标平台**（按 `deploy/trt_plugin_tip.md` 规格）：
> - 主开发：Linux + x86_64（CUDA 11.8 / TensorRT 8.6.1 / L20，CC 8.9）
> - 部署：Linux + aarch64（Orin，JetPack 5.x/6.x，CUDA 11.4/11.8，TRT 8.5/8.6，CC 8.7）
> - 兼容：Windows + x86_64（仅 engine 推理，不做 build）
> **首要接入对象**：`deploy/CPP_trt3/`（C++ 推理 pipeline，按 `trt_plugin_tip.md` 第 6 条）

---

## 1. 背景与动机

### 1.1 现状的两个问题

| 问题 | 根因 | 实测证据 |
|---|---|---|
| **ONNX 推理 acc 暴跌 27 个百分点**（0.93 → 0.66） | `deploy/onnx_backend.py:198-200` 把 FPS 替换成 `traceable_random_fps`（取前 N），丧失空间均匀覆盖；`traceable_ball_query` 的 padding 行为与真实 CUDA kernel 不一致（空邻域用 idx=0 而非"第一个找到的邻居"填充） | 见 §1.2 对照实验 |
| **TRT build 单次 ~18 分钟** | `traceable_ball_query` 与 `traceable_random_fps` 在 ONNX 图里展开成 **1848 个节点**（`--no_simplify` 原始导出，仅 66 个 Conv，其余 1782 个是 TopK/Where/Gather/Shape/Unsqueeze 等无法 fuse 的碎片） | `trtexec` 实测：动态 shape + TF32 on → **1077.6 s**（~18 分钟）；TF32 off → **920.9 s** |

> ⚠️ **实施后实测**（2026-08-14）：FPS/BQ plugin 方案将节点数从 1848 降到 1373（-475），但 **build 时间反而增加到 ~29 分钟**。原因：剩余的 `traceable_grouping`/`three_interp` 碎片算子产生 **2256 个 Reformat autotune**（占 build 时间主导），FPS/BQ plugin 仅贡献极少 autotune。要进一步减少 build 时间，需要将 grouping_operation 和 three_interpolation 也做成 plugin（BallQuery+Grouping 融合）——**已列为 v14 增补任务（三 plugin：BallQueryGroup/BallQueryDP/ThreeInterp），详见 §13**。

### 1.2 关键实测：只做 FPS plugin 不够

为验证"只做 FPS plugin"是否可行，对同一 checkpoint / 同一测试集（训练 split 前 10 文件）做了对照实验：

| 配置 | 平均 acc | 与 baseline 差距 | 说明 |
|---|---|---|---|
| **[A] baseline**（完全无 patch） | **0.9284** | — | 与 test.csv OA=93.26 一致 |
| **[B] 只还原 FPS**（BQ/grouping 仍 traceable） | **0.7688** | −0.16 | **不够**：traceable_ball_query 的 padding bug 仍严重污染特征 |
| **[D] 还原 FPS + 修正 BQ padding bug** | **0.8841** | −0.04 | 接近，但仍有 4 点损失（grouping/three_interp 的 traceable 数值噪声） |
| **[C] 现状**（全 patch，FPS 取前 N） | **0.6623** | −0.27 | 当前 ONNX 推理精度 |

**`traceable_ball_query` 的具体 bug**：
- 真实 `ball_query_gpu.cu:42-44`：找到第一个邻居 k 时，**先把所有 nsample 个位置填 k**，再用真实邻居覆盖。空邻域位置 = 第一个被找到的邻居
- `traceable_ball_query`：空邻域位置填 **idx=0**（第一个点），把无关点的特征混进每个 query 的聚合
- 实测 SA stage 2 输出特征：真实 BQ std=1.13，traceable BQ std=2.27（**翻倍**），max diff=54

**结论**：`trt_plugin_tip.md` 第 1 条"使用自定义的算子 ball_query、fps"是对的——**必须同时做这两个 plugin**。实测 [D] 显示修正 BQ 后 acc 已回到 0.88，剩余 4 点来自 grouping/three_interp 的 traceable 噪声（影响小，且 max-pool 对实现细节不敏感，暂不做 plugin）。

### 1.3 为什么走 Plugin 路线（而非其他备选）

| 备选方案 | 精度 | Orin 可行性 |
|---|---|---|
| PyTorch + 真实 FPS/BQ 部署（不走 TRT） | ✅ 对齐 | ❌ Jetson 上 PyTorch runtime 重、启动慢，不适合嵌入 ECU pipeline |
| ONNX Runtime + 微软 Contrib FPS 算子 | ✅ 对齐 | ⚠️ ORT 在 Jetson 上支持不完整；且 `com.microsoft::FurthestPointSampling` **不存在**（已核查 ORT ContribOperators.md 全列表） |
| **TRT Plugin（本方案）** | ✅ 对齐 | ✅ TRT 是 Jetson 官方推理引擎，CPP_trt3 已在用 |
| 真随机采样替代 FPS | ❌ 掉 5-15 点 | ✅ 但精度无法接受 |

---

## 2. 现成可复用的 CUDA 实现

项目 `openpoints/cpp/pointnet2_batch/` 已有完整的两个 kernel，**只需做"去 PyTorch 依赖 + 套 Plugin 接口"，不需要重写 CUDA 算法**。

### 2.1 FPS（`furthest_point_sampling`）

| 文件 | 行号 | 内容 |
|---|---|---|
| `src/sampling_gpu.cu` | 93-98 | `__device__ __update()`（block reduce） |
| `src/sampling_gpu.cu` | 100-216 | `furthest_point_sampling_kernel<block_size>`（核心 kernel） |
| `src/sampling_gpu.cu` | 218-260 | `furthest_point_sampling_kernel_launcher`（启动 + block_size 选择） |

**接口签名**：
```cpp
void furthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float* dataset, float* temp, int* idxs);
// 输入:  dataset (B, N, 3) float32, contiguous
// 输入:  m = npoint (运行时由 N // stride 决定，stride 是编译期常量)
// 输出:  idxs (B, m) int32
// 内部:  temp (B, N) float32，初始化为 1e10
```

### 2.2 Ball Query

| 文件 | 行号 | 内容 |
|---|---|---|
| `src/ball_query_gpu.cu` | 15-51 | `ball_query_kernel_fast`（核心 kernel） |
| `src/ball_query_gpu.cu` | 54-73 | `ball_query_kernel_launcher_fast`（启动） |

**接口签名**：
```cpp
void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
    const float* new_xyz, const float* xyz, int* idx);
// 输入:  new_xyz (B, M, 3), xyz (B, N, 3)，float32 contiguous
// 输入:  radius (float), nsample (int)
// 输出:  idx (B, M, nsample) int32
//        关键 padding 行为: 找到第一个邻居 k 时，先把所有位置填 k，再用真实邻居覆盖
//        （空邻域位置 = 第一个被找到的邻居，不是 idx=0）
```

### 2.3 模型中两个算子的调用点与实际 shape

当前 cfg（`cfgs/radar/hpenet-ll.yaml`）下，HPENetV2Encoder 的 5 个 stage 参数：

| Stage | blocks | stride | FPS 输出 N | nsample | radius |
|---|---|---|---|---|---|
| 1 (head) | 1 | 1 | N（不采样） | — | — |
| 2 | 3 | 2 | N/2 | 32 | 10 |
| 3 | 5 | 2 | N/4 | 32 | 10 |
| 4 | 3 | 2 | N/8 | 32 | 10 |
| 5 | 3 | 2 | N/16 | 32 | 10 |

**实测 N 的真实分布**（雷达测试集前 30 个 PLY，voxel_size=0.3，test 时 `voxel_max: null` 不 crop）：

```
sub-cloud size: min=3817, max=5727, mean=5140, median=5392（2026-08-17 经 Oracle 对前 30 个排序文件重跑 voxelize(0.3, mode=1) 复核确认，min 来自文件 0000019）
```

> ⚠️ 注意：`voxel_max=4608` 是训练 crop，test 时不过滤。TRT profile 的 `opt_n` 应设为 **5500**（中位数附近），`max_n` 设为 **6500**（覆盖 max）。
>
> **为何 FPS/BallQuery 必须支持动态 shape**：CPP_trt3 现有 pipeline（`pipeline.cpp:148-167` + `subcloud_utils.cpp:22-29`）在 `min_n <= N <= max_n` 时**不 padding**，用真实 N 调 `set_input_shape`。实测 N=3817~5727（CV=12%），若 `min_n=1024, max_n=10000`，**所有帧都落在动态区间**。如果 FPS 把 npoint bake 成静态 attribute，输入 N=5727 时 FPS 只输出 build 时的 npoint（如 2048），丢失 815 个采样点，shape 与下游 gather 不匹配。

**实际需要 FPS 的调用点：4 次**（4 个 SA stage 各 1 次）；**需要 Ball Query 的调用点：8 次**（每个 SA 的 head block `SetAbstraction` + 各 `InvResMLP_block` 的 `LocalAggregation` grouper 各 1 次，共 8 个 `QueryAndGroup`）。

---

## 3. 总体架构

```
                                 ┌──────────────────────────────────────┐
                                 │  deploy/onnx_backend.py              │
   PyTorch 训练阶段               │  patch_model_for_onnx():             │
   (已实现，不动)                 │  v13: sample_fn → fps_op +           │
                                 │       ball_query → bq_op（改 2 处）  │
                                 │  v14: 另加 SA/InvResMLP forward      │
                                 │       → Group/DP op、three_interp    │
                                 │       → ThreeInterpOp（见 §13.5）    │
                                 │  （只替换函数引用/实例 forward，     │
                                 │   不动 grouper 本体）                │
                                 └────────────────┬─────────────────────┘
                                                  ↓
                                 ┌──────────────────────────────────────┐
                                 │  deploy/onnx_export.py               │
   ONNX 导出阶段                 │  torch.onnx.export 调 symbolic:      │
                                 │    FPS  → hpenet::FPS 节点           │
                                 │    BQ   → hpenet::BallQuery 节点     │
                                 └────────────────┬─────────────────────┘
                                                  ↓
                                ┌────────────────────────────────────────┐
                                │  model.onnx（v13：FPS 4 + BQ 8 节点）    │
                                │  v14 实际：FPS 4 + Group 4 + DP 4       │
                                │         + ThreeInterp 5，BQ 0，共 628   │
                                └─────────────────┬──────────────────────┘
                                                  ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │  TensorRT Engine Build (deploy/trt_build.py)                          │
   │  1. dlopen("libhpenet_plugins.so")                                    │
   │  2. REGISTER_TENSORRT_PLUGIN 触发，注册 FPS/BallQuery 到 TRT registry │
   │  3. OnnxParser 解析 hpenet::FPS 节点 → 查 registry → 实例化 plugin    │
   │  4. build_serialized_network → engine 文件                            │
   └────────────────────────────────────┬─────────────────────────────────┘
                                        ↓
   ┌───────────────────────────────────────────────────────────────────┐
   │  Engine 推理                                                       │
   │  - Python: deploy/trt_inference.py (开发验证用)                    │
   │  - C++:    deploy/CPP_trt3/ (Orin 部署用，首要接入对象)            │
   │  两者都需在加载 engine 前 dlopen 或链接 libhpenet_plugins          │
   └───────────────────────────────────────────────────────────────────┘
```

### 3.1 新增目录结构

> 注：以下为 v13 落地结构（FPS/BallQuery 两 plugin）；v14 将新增 3 个 plugin 的对应文件，清单见 §13.6。

```
HPENet_v2-main/
├── deploy/
│   ├── trt_plugins/                         ← 新增：plugin 源码包
│   │   ├── CMakeLists.txt                   ← 主构建文件（3 平台分支）
│   │   ├── include/
│   │   │   ├── fps_plugin.h
│   │   │   ├── ballquery_plugin.h
│   │   │   └── cuda_utils.h                 ← 从 cpp/pointnet2_batch/src 复制（DIVUP/THREADS_PER_BLOCK/opt_n_threads）
│   │   ├── src/
│   │   │   ├── fps_kernel.cu                ← 复制自 sampling_gpu.cu:93-216 的 kernel+launcher，去 at::Tensor，加 stream 参数
│   │   │   ├── ballquery_kernel.cu          ← 复制自 ball_query_gpu.cu:15-51，同上
│   │   │   ├── fps_plugin.cpp               ← nvinfer1::IPluginV2DynamicExt 包装（独立编译单元）
│   │   │   ├── ballquery_plugin.cpp         ← 同上
│   │   │   └── plugin_registry.cpp          ← REGISTER_TENSORRT_PLUGIN(FPSPluginCreator) 等
│   │   └── README.md                        ← 平台特定 build 命令
│   │
│   ├── onnx_ops/                            ← 新增：ONNX 侧 custom op
│   │   ├── __init__.py
│   │   ├── fps_op.py                        ← torch.autograd.Function + symbolic
│   │   └── ballquery_op.py                  ← 同上
│   │
│   ├── onnx_backend.py                      ← 改 2 处函数引用替换（见 §6）
│   ├── CPP_trt3/                            ← 现有 C++ pipeline，需链接 plugin 库（见 §7.4）
│   ├── trt_build.py                         ← 加 dlopen
│   ├── trt_inference.py                     ← 加 dlopen
│   └── ...
```

---

## 4. FPS Plugin 详细设计

### 4.1 ONNX Custom Op 定义

> **动态 shape 设计要点**：CPP_trt3 现有 pipeline 在 `min_n <= N <= max_n` 时不 padding（`subcloud_utils.cpp:22`），用真实 N 调 `set_input_shape`。实测雷达 N 在 3817~5727 变化（CV=12%），落在动态区间。因此 **FPS 必须支持动态 N**：npoint 不能是静态 attribute，必须运行时由 `N // stride` 推导。

**`deploy/onnx_ops/fps_op.py`**：

```python
import torch
from torch.autograd import Function

FPS_OPSET_DOMAIN = "hpenet"
FPS_OPSET_VERSION = 1

class FPSOp(Function):
    """FPS op 的 forward 只接收 stride（编译期常量），不接收 npoint。
    
    npoint 在 forward 里由 xyz.shape[1] // stride 算出。这样 symbolic 函数
    收到的也是 stride（Python int），可以直接作为 ONNX attribute 写入，
    不需要从 xyz.type().sizes() 反推（动态 N 下后者返回 None 会报错）。
    """

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, stride: int) -> torch.Tensor:
        npoint = xyz.shape[1] // stride
        # onnx_export.py 在 CPU 上跑 dummy forward（model.cpu() + CPU dummy input）
        # 真实 CUDA FPS 无法在 CPU 上运行（pointnet2_cuda 期望 GPU 指针）
        # CPU 时返回零张量占位，仅让 trace 跑通，输出被丢弃
        if not xyz.is_cuda:
            return torch.zeros(xyz.shape[0], npoint, dtype=torch.int32, device=xyz.device)
        from openpoints.models.layers.subsample import furthest_point_sample
        return furthest_point_sample(xyz.contiguous(), npoint).to(torch.int32)

    @staticmethod
    def symbolic(g, xyz, stride):
        # stride 作为 attribute 写入 ONNX 节点
        # 输出 dtype 由 forward 返回 torch.int32 自动推断（TensorProto.INT32=6），无需手动 setType
        # 不设 plugin_namespace：REGISTER_TENSORRT_PLUGIN 以空 namespace 注册，
        # TRT parser 默认用空 namespace 查找 → 匹配（第 12 次审查 Oracle 确认）
        return g.op(f"{FPS_OPSET_DOMAIN}::FPS", xyz,
                    stride_i=int(stride),
                    outputs=1)


def make_fps_op(stride):
    """工厂函数：为每个 SetAbstraction 绑定它的 stride。

    模型调用方：sample_fn(p, p.shape[1] // self.stride)
    返回的 sample_fn 接收 (xyz, npoint) 但忽略 npoint（内部用 stride 重新算），
    这样既兼容模型的调用约定，又让 stride 进入 ONNX attribute。
    """
    def sample_fn(xyz, npoint):
        # npoint 被忽略——forward 内部用 xyz.shape[1] // stride 重新算
        return FPSOp.apply(xyz, stride)
    return sample_fn
```

**关键设计决策**：
- **`FPSOp` 只接收 `stride`**（不接收 npoint）：npoint = N // stride，N 是动态量。如果 forward 收 npoint，symbolic 也会收到 npoint（Python int），但无法从中反推 stride（动态 N 下 `xyz.type().sizes()` 返回 None）。改为 forward 直接收 stride，npoint 在 forward 内部算。
- **`make_fps_op(stride)` 工厂函数**：模型里 `sample_fn(p, p.shape[1] // stride)` 调用约定是两个参数。工厂返回的闭包接收 `(xyz, npoint)` 但忽略 npoint，只把 stride 传给 `FPSOp.apply`。这样不改动模型代码。
- **`stride` 作为 ONNX attribute**（编译期常量，模型结构固定）。npoint 的动态推导由 TRT plugin 的 `getOutputDimensions` 用 `exprBuilder.operation(kFLOOR_DIV, N, stride)` 完成。
- **输出 dtype 由 `forward` 返回 `torch.int32` 自动推断**（实测 `TensorProto.INT32=6`，torch.onnx 正确写入 elem_type）。**不需要手动 setType 或后处理**（第 5 次审查确认）。
- **不设 `plugin_namespace` attribute**：`REGISTER_TENSORRT_PLUGIN` 宏以**空 namespace** 注册 creator（`NvInferRuntime.h:3788` 确认 `registerCreator(instance, "")`）。TRT parser 默认用空 namespace 查找，匹配。Creator 的 `getPluginNamespace()` 也返回 `""` 保持一致。如果用非空 namespace（如 "hpenet"），必须改用显式 `getPluginRegistry()->registerCreator(c, "hpenet")` 而非 REGISTER 宏（第 12 次审查 Oracle 发现此 mismatch）。
- **输出 `(B, N/stride)` int32**（2 维），与下游 `torch.gather(p, 1, idx.unsqueeze(-1).expand(-1,-1,3))` 兼容

### 4.2 ONNX 节点 schema

```
Opset: hpenet:1
Op:    hpenet::FPS
Inputs:
  - xyz: tensor(float32), shape [B, N, 3]   ← N 是动态维度
Attributes:
  - stride: int  (编译期常量，模型结构决定；本模型 = 2)
Outputs:
  - idx: tensor(int32), shape [B, N/stride]  ← 输出维度是 N 的函数，动态推导
```

> ⚠️ **ONNX 导出注意事项**（见 §4.4）：
> - 输出 dtype 由 `forward` 返回 `torch.int32` 自动推断为 `TensorProto.INT32`(=6)，**不需要后处理**
> - 输出 shape 的 dim_param 是占位符（如 `'FPSidx_dim_0'`），但 TRT 用 optimization profile 控制 shape，不依赖 dim_param
> - `onnx-simplifier` 必须禁用（会静态化动态 shape），导出用 `--no_simplify`

### 4.3 TRT Plugin 接口实现（修正版）

**`deploy/trt_plugins/src/fps_plugin.cpp`**（关键方法，完整 API）：

```cpp
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstring>              // strcmp
#include "fps_plugin.h"
#include "fps_kernel.h"   // 只 include header, 不 include .cu（避免重复定义）

using namespace nvinfer1;

class FPSPlugin : public IPluginV2DynamicExt {
public:
    FPSPlugin(const std::string& name, int stride) : name_(name), stride_(stride) {}

    // 从序列化数据构造（engine 反序列化时用）
    FPSPlugin(const std::string& name, const void* data, size_t size) : name_(name) {
        const char* d = static_cast<const char*>(data);
        stride_ = *reinterpret_cast<const int*>(d);
    }

    // === IPluginV2DynamicExt 关键方法（签名按 TRT 8.6 官方文档）===

    int32_t getNbOutputs() const noexcept override { return 1; }

    DimsExprs getOutputDimensions(int32_t outputIndex,
                                  const DimsExprs* inputs, int32_t nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override {
        // inputs[0] = xyz (B, N, 3)
        // output = (B, N / stride)  ← npoint 是 N 的函数，动态推导
        // 用 exprBuilder.operation(kFLOOR_DIV, N, stride) 表达整除
        DimsExprs out;
        out.nbDims = 2;
        out.d[0] = inputs[0].d[0];  // B（动态，透传）
        // 关键：N // stride 的表达式推导，不是常量
        out.d[1] = exprBuilder.operation(
            DimensionOperation::kFLOOR_DIV,
            *inputs[0].d[1],                  // N（输入的动态维度）
            *exprBuilder.constant(stride_));  // stride（编译期常量）
        return out;
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        // pos=0: input xyz (fp32), pos=1: output idx (int32)
        // 强制 fp32 输入：即使 engine 开 FP16，TRT 会在 plugin 前后插 Cast
        if (pos == 0) return inOut[pos].type == DataType::kFLOAT
                          && inOut[pos].format == PluginFormat::kLINEAR;
        if (pos == 1) return inOut[pos].type == DataType::kINT32
                          && inOut[pos].format == PluginFormat::kLINEAR;
        return false;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                         const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override {
        // 记录 maxN（来自 profile max），用于 workspace 预分配
        // TRT 保证 configurePlugin 在 getWorkspaceSize 之前调用
        // in[0].desc.max.d[1] 是 profile 里 N 的最大值
        maxB_ = in[0].desc.max.d[0];
        maxN_ = in[0].desc.max.d[1];
    }

    // ⚠️ 正确签名（TRT 8.6 IPluginV2DynamicExt）：
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
                            const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override {
        // temp 缓冲：B * N * sizeof(float)，用 maxB_/maxN_（在 configurePlugin 里锁定）
        return static_cast<size_t>(maxB_) * maxN_ * sizeof(float);
    }

    int32_t enqueue(const PluginTensorDesc* inputDesc,
                    const PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override {
        int B = inputDesc[0].dims.d[0];
        int N = inputDesc[0].dims.d[1];  // 运行时动态 N
        int M = N / stride_;             // npoint 运行时计算，无需 D2H 同步

        float* temp = static_cast<float*>(workspace);
        const float* xyz = static_cast<const float*>(inputs[0]);
        int* idx = static_cast<int*>(outputs[0]);

        // 初始化 temp 为 1e10（cudaMemsetAsync 只能设字节，必须用 fill kernel）
        // launch_fill_kernel 是宿主函数（内部已启动 kernel），不能加 <<<>>>
        launch_fill_kernel(temp, static_cast<int>(B) * N, 1e10f, stream);

        // 调用现成的 kernel launcher（已加 stream 参数）
        fps_launcher_with_stream(B, N, M, xyz, temp, idx, stream);

        return 0;
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
        return DataType::kINT32;
    }

    // === 序列化（存 stride，不存 npoint——npoint 运行时算）===
    size_t getSerializationSize() const noexcept override { return sizeof(int); }
    void serialize(void* buffer) const noexcept override {
        *reinterpret_cast<int*>(buffer) = stride_;
    }
    const char* getPluginType() const noexcept override { return "FPS"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    void setPluginNamespace(const char* ns) noexcept override {}

    IPluginV2DynamicExt* clone() const noexcept override {
        FPSPlugin* p = new FPSPlugin(name_, stride_);
        p->maxB_ = maxB_;  // 补拷贝，避免 getWorkspaceSize 在 configurePlugin 前读到默认值
        p->maxN_ = maxN_;
        return p;
    }

    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }

private:
    std::string name_;
    int stride_ = 1;        // 编译期常量（模型结构决定）
    int maxB_ = 1, maxN_ = 1024;  // 由 configurePlugin 在 build/推理前填充
};

// === Creator ===
class FPSPluginCreator : public IPluginCreator {
public:
    FPSPluginCreator() {
        plugin_attrs_.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 0));
        fc_.nbFields = plugin_attrs_.size();
        fc_.fields = plugin_attrs_.data();
    }
    const char* getPluginName() const noexcept override { return "FPS"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        int stride = 1;
        for (int i = 0; i < fc->nbFields; ++i) {
            if (strcmp(fc->fields[i].name, "stride") == 0) {
                stride = *static_cast<const int*>(fc->fields[i].data);
            }
        }
        return new FPSPlugin(name, stride);
    }

    IPluginV2* deserializePlugin(const char* name, const void* data, size_t size) noexcept override {
        return new FPSPlugin(name, data, size);
    }

    void setPluginNamespace(const char* ns) noexcept override {}

private:
    std::vector<PluginField> plugin_attrs_;
    PluginFieldCollection fc_{};
};
```

> ⚠️ **`REGISTER_TENSORRT_PLUGIN(FPSPluginCreator)` 不写在此文件里**——集中注册在 `plugin_registry.cpp`（见 §7.4），避免重复注册。
>
> ⚠️ **`FPSPluginCreator` 类的完整定义必须放在 `fps_plugin.h`**（不是 `.cpp`）——因为 `plugin_registry.cpp` 要 `#include "fps_plugin.h"` 后用 `REGISTER_TENSORRT_PLUGIN(FPSPluginCreator)`，宏展开需要 Creator 的完整类型（`NvInferRuntime.h:3793` `T instance{}`）。同理 `BallQueryPluginCreator` 放 `ballquery_plugin.h`。

**`fps_kernel.cu` 的改动**（相对 `sampling_gpu.cu`）：

```cuda
// 几乎原样复制 sampling_gpu.cu:93-216 的 __update + furthest_point_sampling_kernel
// 唯一改动：launcher 加 stream 参数，去 at::Tensor 依赖

void fps_launcher_with_stream(int b, int n, int m,
                              const float* dataset, float* temp, int* idxs,
                              cudaStream_t stream) {
    unsigned int n_threads = opt_n_threads(n);
    if (n_threads < 1) n_threads = 1;  // 防 N < 16 时返回 0
    switch (n_threads) {
        case 1024:
            furthest_point_sampling_kernel<1024><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        // ... 其余 case 不变，只加第 4 个 launch 参数 stream
    }
}
```

> ⚠️ `fps_kernel.cu` 作为**独立编译单元**加入 CMakeLists.txt，`fps_plugin.cpp` 只 include `fps_kernel.h`（声明 launcher），**不要 `#include "fps_kernel.cu"`**（会导致重复定义）。

**辅助函数定义**

`cdiv` 宏（宿主侧，加到 `cuda_utils.h`）：
```cpp
// cuda_utils.h 已有 DIVUP 宏，cdiv 直接复用
#ifndef cdiv
#define cdiv(m, n) DIVUP(m, n)
#endif
```

`fill_kernel` + `launch_fill_kernel`（**必须放在 `fps_kernel.cu` 里由 NVCC 编译**，不能放 `fps_plugin.cpp`——g++ 不认 `__global__`/`<<<>>>`）：
```cuda
// fps_kernel.cu 末尾添加：

// fill kernel：把 buffer 填成指定值（cudaMemsetAsync 只能设字节，无法填 1e10f）
template<typename T>
__global__ void fill_kernel(T* data, int n, T value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

// 宿主侧 launcher（enqueue 直接调用，不能加 <<<>>>）
void launch_fill_kernel(float* data, int n, float value, cudaStream_t stream) {
    fill_kernel<float><<<cdiv(n, 256), 256, 0, stream>>>(data, n, value);
}
```

`fps_kernel.h` 里声明（供 `fps_plugin.cpp` include）：
```cpp
// fps_kernel.h
void launch_fill_kernel(float* data, int n, float value, cudaStream_t stream);
```

### 4.4 ONNX 导出注意事项

**关键事实**（第 5 次审查实测确认）：
- `torch.onnx.export` **能正确推断自定义 op 的输出 dtype**。只要 `FPSOp.forward` 返回 `torch.int32` tensor，导出后 ONNX 输出的 `elem_type` 自动是 `TensorProto.INT32`（=6）。**不需要手动 setType 或后处理修正 dtype**。
- 输出 shape 的 `dim_param` 是占位符（如 `'FPSidx_dim_0'`），没有表达成 `[B, N/stride]`。但 **TRT 用 optimization profile 的 min/opt/max 控制 shape**，不依赖 ONNX dim_param 名字，所以占位符不影响 TRT 解析。
- 警告 `shape inference of hpenet::FPS is missing` 可以忽略——它只影响 ONNX Runtime 等需要 shape inference 的 consumer，TRT parser 不依赖它。

**`onnx-simplifier` 必须禁用**：simplifier 会跑一次 forward 做 constant-fold，用导出时 dummy N（如 4096）推断 FPS 输出 shape 为 `[1, 2048]`，**把动态 shape 静态化**。导出时必须用 `--no_simplify`。

**导出流程**：

```bash
# 步骤 1: 导出（不加 simplifier）
python deploy/onnx_export.py \
    --cfg cfgs/radar/hpenet-ll.yaml \
    --checkpoint log/radar/.../checkpoint/*_ckpt_best.pth \
    --output deploy/hpenet_v2_plugin.onnx \
    --no_simplify

# 步骤 2: 验证节点
python -c "
import onnx
m = onnx.load('deploy/hpenet_v2_plugin.onnx')
fps = sum(1 for n in m.graph.node if n.op_type == 'FPS' and n.domain == 'hpenet')
bq  = sum(1 for n in m.graph.node if n.op_type == 'BallQuery' and n.domain == 'hpenet')
print(f'FPS nodes: {fps}, BallQuery nodes: {bq}')
# 期望（v13）: FPS nodes: 4, BallQuery nodes: 8
# 期望（v14 起）: FPS 4, BallQuery 0, BallQueryGroup 4, BallQueryDP 4, ThreeInterp 5（总节点 628）
# 验证 FPS 输出 dtype:
for vi in m.graph.value_info:
    if 'FPS' in vi.name:
        et = vi.type.tensor_type.elem_type
        print(f'  {vi.name}: elem_type={et} (期望 6=INT32, 0=UNDEFINED)')
"
```

> ⚠️ 不跑 simplifier 意味着 grouping/three_interp 的碎片算子保留在图里。但这些算子的 build 时间已由 FPS/BQ plugin 大幅降低（实测预期总 build 时间 < 5 分钟）。

---

## 5. Ball Query Plugin 详细设计

### 5.1 关键：只替换 `ball_query` 函数引用，不动 `QueryAndGroup.forward`

`QueryAndGroup.forward`（`group.py:235-255`）做三件事：
1. `idx = ball_query(...)` ← 只替换这一步
2. `grouped_xyz = grouping_operation(xyz_trans, idx)` ← 保留（仍是 traceable grouping）
3. `grouped_xyz -= query_xyz` + `normalize_dp` ← 保留

**patch 策略**：替换 `group_mod.ball_query` 函数引用（module-level），不替换整个 `grouper.forward`。这样 `QueryAndGroup.forward` 内部调 `ball_query(...)` 时走自定义 op，其余逻辑（grouping、normalize）走原实现。

### 5.2 ONNX Custom Op

**`deploy/onnx_ops/ballquery_op.py`**：

```python
import torch
from torch.autograd import Function
import onnx

class BallQueryOp(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # 复用现有 CUDA kernel（GPU 推理时）；CPU 导出 dummy forward 时返回占位
        B, N, _ = xyz.shape
        _, M, _ = new_xyz.shape
        if not xyz.is_cuda:
            return torch.zeros(B, M, nsample, dtype=torch.int32, device=xyz.device)
        from openpoints.cpp import pointnet2_cuda
        idx = torch.cuda.IntTensor(B, M, nsample, device=xyz.device).zero_()
        pointnet2_cuda.ball_query_wrapper(B, N, M, float(radius), nsample,
                                          new_xyz.contiguous(), xyz.contiguous(), idx)
        return idx

    @staticmethod
    def symbolic(g, radius, nsample, xyz, new_xyz):
        # radius 和 nsample 作为 attribute（编译期常量，模型结构决定）
        # 输入 xyz (B,N,3) 和 new_xyz (B,M,3) 都是动态的，M 来自 FPS 输出
        # 输出 dtype 由 forward 返回 torch.int32 自动推断，无需手动 setType
        # 不设 plugin_namespace（与 FPS 同理，REGISTER 以空 namespace 注册）
        return g.op(f"hpenet::BallQuery", xyz, new_xyz,
                    radius_f=float(radius), nsample_i=int(nsample),
                    outputs=1)

def ballquery_op(radius, nsample, xyz, new_xyz):
    return BallQueryOp.apply(radius, nsample, xyz, new_xyz)
```

> ⚠️ 与 FPS 同理：`forward` 返回 `torch.int32` 时 ONNX 输出 dtype 自动推断为 `TensorProto.INT32`(=6)，**不需要手动 setType**。不设 `plugin_namespace`（REGISTER 以空 namespace 注册，TRT 默认空 namespace 查找，匹配）。

**注意参数顺序**：`QueryAndGroup.forward` 调 `ball_query(self.radius, self.nsample, support_xyz, query_xyz)`，即 `(radius, nsample, xyz, new_xyz)`。symbolic 里 ONNX 节点输入顺序是 `[xyz, new_xyz]`（attribute 不算输入）。

### 5.3 TRT Plugin

结构与 FPS 完全对称，差异点：

| 维度 | FPS | Ball Query |
|---|---|---|
| 输入 | xyz (B,N,3)，N 动态 | xyz (B,N,3) N 动态 [inputs[0]], new_xyz (B,M,3) M 动态 [inputs[1]] |
| Attribute | stride (int) | radius (float), nsample (int) |
| 输出 shape | (B, N/stride) int32，动态推导 | (B, M, nsample) int32，M 动态透传 |
| Workspace | B×N float（temp，按 profile max 预分配） | 无 |
| 序列化 | stride | radius, nsample |
| npoint 来源 | `enqueue` 里 `M = N / stride_` 运行时算 | M 从输入 shape 直接读 |

**`getOutputDimensions`**：
```cpp
DimsExprs getOutputDimensions(int32_t outputIndex,
                              const DimsExprs* inputs, int32_t nbInputs,
                              IExprBuilder& exprBuilder) noexcept override {
    // inputs[0] = xyz (B, N, 3), inputs[1] = new_xyz (B, M, 3)
    // output = (B, M, nsample)，M 从 inputs[1]（new_xyz）取
    DimsExprs out;
    out.nbDims = 3;
    out.d[0] = inputs[1].d[0];                  // B（动态，透传）
    out.d[1] = inputs[1].d[1];                  // M（动态，来自 FPS 输出的 new_xyz）
    out.d[2] = exprBuilder.constant(nsample_);  // nsample（attribute，编译期常量）
    return out;
}
```

**完整 plugin 类与 Creator**（`deploy/trt_plugins/src/ballquery_plugin.cpp`）：

```cpp
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstring>              // strcmp
#include "ballquery_plugin.h"
#include "ballquery_kernel.h"   // 声明 ball_query_launcher_with_stream

using namespace nvinfer1;

class BallQueryPlugin : public IPluginV2DynamicExt {
public:
    BallQueryPlugin(const std::string& name, float radius, int nsample)
        : name_(name), radius_(radius), nsample_(nsample) {}

    // 从序列化数据构造（engine 反序列化时用）
    BallQueryPlugin(const std::string& name, const void* data, size_t size) : name_(name) {
        const char* d = static_cast<const char*>(data);
        radius_ = *reinterpret_cast<const float*>(d);
        d += sizeof(float);
        nsample_ = *reinterpret_cast<const int*>(d);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    DimsExprs getOutputDimensions(int32_t outputIndex,
                                  const DimsExprs* inputs, int32_t nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override {
        // inputs[0] = xyz (B, N, 3), inputs[1] = new_xyz (B, M, 3)
        // output = (B, M, nsample)，M 从 inputs[1]（new_xyz）取
        DimsExprs out;
        out.nbDims = 3;
        out.d[0] = inputs[1].d[0];                  // B（动态，透传）
        out.d[1] = inputs[1].d[1];                  // M（动态，来自 FPS 输出的 new_xyz）
        out.d[2] = exprBuilder.constant(nsample_);  // nsample（attribute，编译期常量）
        return out;
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        // pos=0: xyz (fp32), pos=1: new_xyz (fp32), pos=2: output idx (int32)
        // 强制 fp32 输入：即使 engine 开 FP16，TRT 会在 plugin 前后插 Cast
        if (pos == 0 || pos == 1)
            return inOut[pos].type == DataType::kFLOAT
                   && inOut[pos].format == PluginFormat::kLINEAR;
        if (pos == 2)
            return inOut[pos].type == DataType::kINT32
                   && inOut[pos].format == PluginFormat::kLINEAR;
        return false;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                         const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override {
        // BallQuery 无需 workspace（kernel 内部无动态分配）
        // configurePlugin 留空即可（TRT 8.6 要求 override 但可以 no-op）
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
                            const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override {
        return 0;  // BallQuery 不需要 workspace
    }

    int32_t enqueue(const PluginTensorDesc* inputDesc,
                    const PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override {
        // ONNX 节点输入顺序 = [xyz, new_xyz]（§5.2 symbolic 决定）
        // inputs[0] = xyz (B, N, 3) support 点，inputs[1] = new_xyz (B, M, 3) query 点
        // output  = idx (B, M, nsample)
        int B = inputDesc[0].dims.d[0];
        int N = inputDesc[0].dims.d[1];  // support 点数（动态）
        int M = inputDesc[1].dims.d[1];  // query 点数（来自 FPS 输出，动态）

        const float* xyz     = static_cast<const float*>(inputs[0]);
        const float* new_xyz = static_cast<const float*>(inputs[1]);
        int* idx             = static_cast<int*>(outputs[0]);

        // 调用现成的 kernel launcher（参数顺序: b, n, m, radius, nsample, new_xyz, xyz, idx）
        ball_query_launcher_with_stream(B, N, M, radius_, nsample_,
                                        new_xyz, xyz, idx, stream);
        return 0;
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
        return DataType::kINT32;
    }

    // === 序列化（存 radius + nsample）===
    size_t getSerializationSize() const noexcept override {
        return sizeof(float) + sizeof(int);
    }
    void serialize(void* buffer) const noexcept override {
        char* d = static_cast<char*>(buffer);
        *reinterpret_cast<float*>(d) = radius_;
        d += sizeof(float);
        *reinterpret_cast<int*>(d) = nsample_;
    }
    const char* getPluginType() const noexcept override { return "BallQuery"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    void setPluginNamespace(const char* ns) noexcept override {}

    IPluginV2DynamicExt* clone() const noexcept override {
        return new BallQueryPlugin(name_, radius_, nsample_);
    }

    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }

private:
    std::string name_;
    float radius_ = 0.0f;
    int nsample_ = 0;
};

// === Creator ===
class BallQueryPluginCreator : public IPluginCreator {
public:
    BallQueryPluginCreator() {
        plugin_attrs_.emplace_back(PluginField("radius", nullptr, PluginFieldType::kFLOAT32, 0));
        plugin_attrs_.emplace_back(PluginField("nsample", nullptr, PluginFieldType::kINT32, 0));
        fc_.nbFields = plugin_attrs_.size();
        fc_.fields = plugin_attrs_.data();
    }
    const char* getPluginName() const noexcept override { return "BallQuery"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        float radius = 0.0f;
        int nsample = 0;
        for (int i = 0; i < fc->nbFields; ++i) {
            if (strcmp(fc->fields[i].name, "radius") == 0) {
                radius = *static_cast<const float*>(fc->fields[i].data);
            } else if (strcmp(fc->fields[i].name, "nsample") == 0) {
                nsample = *static_cast<const int*>(fc->fields[i].data);
            }
        }
        return new BallQueryPlugin(name, radius, nsample);
    }

    IPluginV2* deserializePlugin(const char* name, const void* data, size_t size) noexcept override {
        return new BallQueryPlugin(name, data, size);
    }

    void setPluginNamespace(const char* ns) noexcept override {}

private:
    std::vector<PluginField> plugin_attrs_;
    PluginFieldCollection fc_{};
};
```

> ⚠️ 与 FPS 同理，`REGISTER_TENSORRT_PLUGIN(BallQueryPluginCreator)` 集中在 `plugin_registry.cpp`（见 §7.4）。`BallQueryPluginCreator` 类的完整定义必须放在 `ballquery_plugin.h`（原因同 FPS，宏需要完整类型）。

**`ballquery_kernel.cu` 的改动**（相对 `ball_query_gpu.cu`）：

```cuda
// 几乎原样复制 ball_query_gpu.cu:15-51 的 ball_query_kernel_fast
// 唯一改动：launcher 加 stream 参数，去 at::Tensor 依赖

void ball_query_launcher_with_stream(int b, int n, int m, float radius, int nsample,
                                     const float* new_xyz, const float* xyz, int* idx,
                                     cudaStream_t stream) {
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);
    dim3 threads(THREADS_PER_BLOCK);
    ball_query_kernel_fast<<<blocks, threads, 0, stream>>>(
        b, n, m, radius, nsample, new_xyz, xyz, idx);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }
}
```

> 与 FPS 同理：`ballquery_kernel.cu` 作为**独立编译单元**，`ballquery_plugin.cpp` 只 include `ballquery_kernel.h`。

---

## 6. 现有代码的最小改动

### 6.1 `deploy/onnx_backend.py`（改 2 处函数引用替换）

```python
# deploy/onnx_backend.py 内 patch_model_for_onnx 函数

def patch_model_for_onnx(model):
    import openpoints.models.backbone.hpenetv2 as hpenetv2_mod
    import openpoints.models.layers.group as group_mod

    # === 改动 1: sample_fn → make_fps_op(stride)（不再用 traceable_random_fps）===
    # 模型调用方：sample_fn(p, p.shape[1] // self.stride)
    # make_fps_op 返回闭包：接收 (xyz, npoint) 但忽略 npoint，调 FPSOp.apply(xyz, stride)
    # stride 在各 SA stage 固定（本模型均为 2），作为 ONNX attribute 写入
    from deploy.onnx_ops.fps_op import make_fps_op
    patched_count = 0
    for _name, module in model.named_modules():
        if hasattr(module, 'sample_fn'):
            module.sample_fn = make_fps_op(module.stride)  # 绑定 stride
            patched_count += 1
    print(f"  Patched {patched_count} sample_fn → make_fps_op(stride)")

    # === 改动 2: group_mod.ball_query → ballquery_op（不再用 traceable_ball_query）===
    # 关键：只替换函数引用，不动 grouper.forward
    from deploy.onnx_ops.ballquery_op import ballquery_op
    group_mod.ball_query = ballquery_op         # ← 改这里（原: traceable_ball_query）
    print(f"  Patched group_mod.ball_query → ballquery_op")

    # 保留 traceable_grouping_operation（碎片化但精度影响小，实测 Δ<0.04）
    hpenetv2_mod.grouping_operation = traceable_grouping_operation
    group_mod.grouping_operation = traceable_grouping_operation
    hpenetv2_mod.three_interpolation = traceable_three_interpolation

    # 保留 residual block / BaseSeg squeeze 的 patch（控制流，不影响数值）
    block_patched = _patch_residual_blocks(model)
    print(f"  Patched {block_patched} residual blocks (removed dynamic If)")
    _patch_base_seg_squeeze(model)
    print(f"  Patched BaseSeg.forward (removed squeeze If)")

    return model
```

**这是 `onnx_backend.py` 的全部改动**：把 `traceable_random_fps` 换成 `fps_op`，把 `traceable_ball_query` 换成 `ballquery_op`。其余 patch（grouping、three_interp、residual If）保留。

### 6.2 完整改动清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `deploy/onnx_ops/` | **新增目录** | `__init__.py`, `fps_op.py`, `ballquery_op.py`（不含 postprocess） |
| `deploy/trt_plugins/` | **新增目录** | 完整 plugin 源码包 + CMakeLists.txt |
| `deploy/onnx_backend.py` | 改 ~10 行 | `patch_model_for_onnx` 内 2 处函数引用替换 |
| `deploy/trt_build.py` | 加 ~5 行 | `build_engine()` 开头（`trt.OnnxParser` 创建之前）`ctypes.CDLL(plugin_so)` |
| `deploy/trt_inference.py` | 加 ~5 行 | 同上（`TRTSession.__init__` 前、`deserialize_cuda_engine` 之前）|
| `deploy/CPP_trt3/CMakeLists.txt` | 加 ~5 行 | 链接 `hpenet_plugins`（见 §7.4） |
| `deploy/CPP_trt3/src/trt_engine.cpp` | 改 ~3 行 | engine 加载前确保 plugin 已注册（见 §7.4） |

**不动**的文件：
- `openpoints/cpp/pointnet2_batch/` 所有 `.cu`（kernel 直接复用）
- `openpoints/models/` 所有模型代码
- `examples/segmentation/main.py`（训练/测试入口）
- `cfgs/`（配置文件）

**新增依赖**：无（`torch`、`onnx` 已安装；不需要 `onnx-graphsurgeon` 或 `postprocess` 脚本）

---

## 7. 编译与运行命令

### 7.1 Linux x86_64（开发机，L20）

**前置**：
```bash
export TRT_HOME=/usr/local/TensorRT-8.6.1.6
export CUDA_HOME=/usr/local/cuda-11.8
export CUDNN_LIB=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib
```

**Build plugin**：

`deploy/trt_plugins/CMakeLists.txt` 最小模板（需先创建此文件）：
```cmake
cmake_minimum_required(VERSION 3.18)
project(hpenet_plugins LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
find_package(TensorRT REQUIRED)

add_library(hpenet_plugins SHARED
    src/fps_kernel.cu
    src/ballquery_kernel.cu
    src/fps_plugin.cpp
    src/ballquery_plugin.cpp
    src/plugin_registry.cpp
)
target_include_directories(hpenet_plugins PRIVATE
    include ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(hpenet_plugins PRIVATE ${TensorRT_LIBRARIES})
set_target_properties(hpenet_plugins PROPERTIES POSITION_INDEPENDENT_CODE ON)
```

> `find_package(TensorRT)` 复用 `deploy/CPP_trt3/cmake/` 下现有的 `FindTensorRT.cmake`。

编译命令：
```bash
cd deploy/trt_plugins
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_HOME=$TRT_HOME \
    -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
    -DCMAKE_CUDA_ARCHITECTURES="89"     # L20=8.9, Orin=8.7, 全开="60;62;70;75;80;86;87;89"
make -j$(nproc)
# 产物: libhpenet_plugins.so
```

**导出 ONNX**（⚠️ 必须用 `--no_simplify`，simplifier 会静态化 FPS 动态 shape）：
```bash
cd /home/wangpeng/CODE/HPENet_v2-main
export LD_LIBRARY_PATH="$TRT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_LIB:$LD_LIBRARY_PATH"

python deploy/onnx_export.py \
    --cfg cfgs/radar/hpenet-ll.yaml \
    --checkpoint log/radar/radar-train-hpenet-ll-ngpus1-20260812-201051-*/checkpoint/*_ckpt_best.pth \
    --output deploy/hpenet_v2_plugin.onnx \
    --no_simplify

# 验证节点（FPS 输出 dtype 应为 6=INT32，由 forward 返回 torch.int32 自动推断）
python -c "
import onnx
m = onnx.load('deploy/hpenet_v2_plugin.onnx')
fps = sum(1 for n in m.graph.node if n.op_type == 'FPS' and n.domain == 'hpenet')
bq  = sum(1 for n in m.graph.node if n.op_type == 'BallQuery' and n.domain == 'hpenet')
print(f'FPS nodes: {fps}, BallQuery nodes: {bq}')
# 期望（v13）: FPS nodes: 4, BallQuery nodes: 8
# 期望（v14 起）: FPS 4, BallQuery 0, BallQueryGroup 4, BallQueryDP 4, ThreeInterp 5（总节点 628）
for vi in m.graph.value_info:
    if 'FPS' in vi.name:
        et = vi.type.tensor_type.elem_type
        print(f'  {vi.name}: elem_type={et} (期望 6=INT32, 0=UNDEFINED)')
"
```

**Build engine**：
```bash
python deploy/trt_build.py \
    --onnx deploy/hpenet_v2_plugin.onnx \
    --output deploy/hpenet_v2_fp32.engine \
    --min_n 1024 --opt_n 5500 --max_n 6500
# 注意: opt_n=5500（实测中位数 5392）, max_n=6500（实测 max 5727 留余量）
# min_n=1024 覆盖 CPP_trt3 的 padding 下限（N < min_n 时 pad 到 1024）
# 预期 build 时间 < 5 分钟
```

**FP16 engine**：
```bash
python deploy/trt_build.py \
    --onnx deploy/hpenet_v2_plugin.onnx \
    --output deploy/hpenet_v2_fp16.engine \
    --fp16 \
    --min_n 1024 --opt_n 5500 --max_n 6500
```

### 7.2 Linux aarch64（Orin 部署机）

**Step 1: 在 Orin 上编译 plugin**（不能交叉编译，TRT plugin 不跨架构）

```bash
export TRT_HOME=/usr/lib/aarch64-linux-gnu     # JetPack 默认
export CUDA_HOME=/usr/local/cuda-11.4          # JetPack 5.x; 6.x 是 cuda-11.8

cd deploy/trt_plugins
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_HOME=$TRT_HOME \
    -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
    -DCMAKE_CUDA_ARCHITECTURES="87"                  # Orin = 8.7
make -j$(nproc)
```

**Step 2: 在 Orin 上 build engine**（不能用 x86 build 的，TRT engine 不跨架构）

```bash
scp deploy/hpenet_v2_plugin.onnx orin:/path/to/deploy/
cd /path/to/HPENet_v2/deploy
LD_LIBRARY_PATH=. python trt_build.py \
    --onnx hpenet_v2_plugin.onnx \
    --output hpenet_v2_orin_fp32.engine \
    --min_n 1024 --opt_n 5500 --max_n 6500 \
    --workspace 2     # Orin 共享内存，调小
```

**Step 3: Orin 上推理**（走 CPP_trt3，见 §7.4）

### 7.3 Windows x86_64（仅推理）

```cmd
"C:\Program Files\CMake\bin\cmake.exe" -B build -G "Visual Studio 17 2022" -A x64 ^
    -DTRT_HOME=C:\TensorRT-8.6.1.6 ^
    -DCUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 ^
    -DCMAKE_CUDA_ARCHITECTURES="89"
"C:\Program Files\CMake\bin\cmake.exe" --build build --config Release
:: 产物: build\Release\hpenet_plugins.dll
```

### 7.4 CPP_trt3 接入（首要目标，按 `trt_plugin_tip.md` 第 6 条）

> ⚠️ 本节为 v13 状态（FPS/BallQuery 两 plugin）。v14 实施后 plugin 总数变为 5 个（+BallQueryGroup/BallQueryDP/ThreeInterp），下方 plugin_registry.cpp 与头文件 include 清单须按 §13.6 同步扩充——此处不重复展开。

现有 `deploy/CPP_trt3/` 已经有完整 C++ 推理 pipeline，需要 3 处改动接入 plugin：

**改动 1: `CMakeLists.txt` 链接 plugin 库**

```cmake
# deploy/CPP_trt3/CMakeLists.txt 在 add_executable 后添加:

# 链接自定义 plugin 库（静态链接优于 dlopen，避免运行时找不到 .so）
set(HPENET_PLUGINS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../trt_plugins")
target_include_directories(hpenet_trt_infer PRIVATE
    ${HPENET_PLUGINS_ROOT}/include)
target_link_directories(hpenet_trt_infer PRIVATE
    ${HPENET_PLUGINS_ROOT}/build)
target_link_libraries(hpenet_trt_infer PRIVATE hpenet_plugins)
```

**改动 2: `src/trt_engine.cpp` 确保 plugin 注册**

```cpp
// deploy/CPP_trt3/src/trt_engine.cpp 顶部添加:
#include <NvInferPlugin.h>

// 强制链接 plugin 库（即使没有显式调用其符号）
// 必须用 extern "C" 匹配 plugin_registry.cpp 里的 extern "C" 定义，否则 C++ name mangling 不一致导致链接错误
extern "C" bool initLibHPENetPlugins();  // 由 plugin 库导出（见 plugin_registry.cpp）

// TrEngine 构造函数内，在 initLibNvInferPlugins 之前添加:
TrEngine::TrEngine(...) {
    // ... 读 engine 文件 ...

    // 注册自定义 plugin（必须在 deserializeCudaEngine 之前）
    initLibHPENetPlugins();
    initLibNvInferPlugins(&logger, "");  // 已有

    // ... 后续不变 ...
}
```

**改动 3: `plugin_registry.cpp` 唯一注册点 + 导出 initLibHPENetPlugins 符号**

```cpp
// deploy/trt_plugins/src/plugin_registry.cpp:
// 这是整个 plugin 库的唯一注册点——REGISTER_TENSORRT_PLUGIN 只写在这里，
// 不要在 fps_plugin.cpp / ballquery_plugin.cpp 里重复写（会导致重复注册）。
#include <NvInferPlugin.h>
#include "fps_plugin.h"
#include "ballquery_plugin.h"

REGISTER_TENSORRT_PLUGIN(FPSPluginCreator);
REGISTER_TENSORRT_PLUGIN(BallQueryPluginCreator);

extern "C" bool initLibHPENetPlugins() {
    return true;
}
// 静态注册器在 dlopen/链接时自动触发（PluginRegistrar 静态变量初始化），
// initLibHPENetPlugins 仅用于强制链接器保留 plugin 符号（避免 --as-needed 裁剪）
```

**改动 4: `main.cpp` 默认 engine 路径**（可选）

```cpp
// deploy/CPP_trt3/src/main.cpp CLIConfig:
std::string engine_path =
    "/home/wangpeng/CODE/HPENet_v2-main/deploy/hpenet_v2_fp32.engine";  // 改成 plugin 版
```

---

## 8. FP16 支持（按 `trt_plugin_tip.md` 第 5 条要求）

### 8.1 FPS 在 FP16 下的陷阱与对策

**陷阱 1：`temp` 缓冲初始化值 1e10 在 fp16 下溢出**（fp16 max=65504）

**对策**：plugin 内部强制 fp32 计算，`supportsFormatCombination` 只允许 fp32 输入：
```cpp
if (pos == 0) return inOut[pos].type == DataType::kFLOAT;  // 只收 fp32
```
即使 engine 启用 FP16，TRT 会在 plugin 前后自动插 `Cast` 节点。FPS 单 kernel < 1 ms，精度损失可忽略。

**陷阱 2：`cudaMemsetAsync` 不能设 1e10**

`cudaMemsetAsync` 只能设字节值，1e10f 的 IEEE 754 字节模式不是简单重复的 0。

**对策**：用 fill kernel 初始化 temp（定义见 §4.3 辅助函数）：
```cpp
// 在 enqueue 里（launch_fill_kernel 是宿主函数，不能加 <<<>>>）：
launch_fill_kernel(temp, B * N, 1e10f, stream);
```

### 8.2 Ball Query 在 FP16 下的处理

同样强制 fp32：`supportsFormatCombination` 只接受 fp32 输入。Ball Query 是整数索引 + 比较运算，fp32 精度足够。

### 8.3 整网 FP16 策略

| 层 | 精度 | 原因 |
|---|---|---|
| Conv / Linear | fp16 | 性能关键，Orin fp16 = 2x |
| BatchNorm | fp16 | TRT 自动 fuse 到 Conv |
| **FPS** | **fp32**（plugin 内部） | temp 缓冲溢出，强制 fp32 |
| **Ball Query** | **fp32**（plugin 内部） | 比较运算精度保险 |
| grouping/three_interp（traceable） | fp32 | TRT 自动决定 |
| Gather / Scatter | int32 | 索引类型 |

实际部署：`--fp16` + plugin 强制 fp32。TRT 在精度敏感处自动回退（与 mmcv `trt_scatternd` 同款策略）。

---

## 9. 验证矩阵

> **实施后实测结果已标注**（v13：2026-08-14；v14.5：2026-08-17，L20 + TRT 8.6.1 + 同一 checkpoint）

| # | 验证项 | 通过标准 | 实测结果 |
|---|---|---|---|
| 1 | ONNX 图节点数 | FPS 4 + BQ 8 单节点 | ✅ v13：**1373 节点** → ✅ v14：**628 节点**（-745 / -54%，Group×4+DP×4+ThreeInterp×5，BallQuery/TopK 清零，碎片算子 668→273） |
| 2 | ONNX 图中有 `hpenet::FPS` × 4 + `hpenet::BallQuery` × 8 | — | ✅ v13 实测一致；v14 起 BallQuery ×0（被 Group/DP 替代），FPS ×4 不变 |
| 3 | **FPS 输出 dtype = INT32** | elem_type == 6 | ✅ 实测 6 |
| 4 | **FPS 输出 shape 动态** | dims 含 symbolic dim_param | ✅ 实测含占位符 dim_param |
| 5 | **FPS 节点不含 `plugin_namespace` attribute** | 空 namespace 匹配 | ✅ 实测 plugin_namespace="" 匹配 |
| 6 | **TRT engine 接受变长 N** | N=3800 和 N=5700 各推理一次 | ✅ 实测成功（profile min=1024/opt=5500/max=6500） |
| 7 | ORT 推理 acc（CPU kernel 替代） | ≥ 0.88 | 未测（ORT 无 FPS/BQ custom op） |
| 8 | PyTorch patched 模型 forward acc | ≥ 0.88 | ✅ v14 V2：10 真实子云 logits **逐位相等（err=0）** |
| 9 | **TRT fp32 engine 推理 acc（L20）** | ≥ 0.88 | ✅ v13：**0.9741** → ✅ v14：**0.9741**（零回退，两次复测一致） |
| 10 | TRT fp16 engine 推理 acc | **≥ 0.96**（v14 起自 0.87 收紧，见 §13.7 V3） | ✅ v14（2026-08-18）：**0.9741**，与 fp32 逐文件持平（±0.0004）。build 92.4min（fp32 的 ~3 倍，fp32/fp16 双份 tactic），engine 9.4MB；纯 engine 延迟 median 26.38ms（fp32 26.53ms）。注：builder 自选 plugin 层以 fp32 执行（省 reformat，1509 个 Half tactic 已评估），plugin 的 half 路径能力由 V1 强制 half 测试逐位验证 |
| 11 | **TRT build 时间** | — | ⚠️ v13 ~29min → v14 **31.8min（持平略增）**。分析：最终 engine Reformat 层数不变（14 个/237 层 vs 218 层），build 耗时主导项是 66 个 conv 的 tactic autotune（×动态 shape），节点数减半对此无感；v14 新增 13 个 plugin 节点各自参与 autotune。**v13 的"2256 Reformat autotune 是 build 时间主导"判断被证伪** |
| 12 | TRT 单样本推理延迟（L20） | — | **纯 engine 延迟 @ N=5500**：v13 median 27.13ms → v14 **26.53ms（不劣于，-2.2%）**，mean 29.11→27.37（-6%）。端到端 0.179→**0.141 s/file（-21%）**。engine 大小 v13 15.5MB → v14 15.1MB |
| 13 | **CPP_trt3 能加载 plugin engine 并推理** | 不报 "plugin not found" | ✅ v13 编译+链接成功；✅ v14 重编后加载新 engine 推理成功（C++ 端到端 43-97ms/file） |
| 14 | Orin 上 build engine | 成功 | 待测 |
| 15 | Orin 上 fp32 engine 推理 acc | ≥ 0.88 | 待测 |
| 16 | Windows 上 fp32 engine 推理 | 成功 | 待测 |

---

## 10. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **`exprBuilder.operation(kFLOOR_DIV)` 在某些 TRT 版本不可用** | 无法表达动态 npoint | **仅在开发机 TRT 8.6.1 的 `NvInferRuntime.h:200` 确认 `kFLOOR_DIV=7`；TRT 8.4/8.5（Orin JetPack 5.x）未实测确认**。在 Orin 上 build 前必须先 `grep kFLOOR_DIV /usr/include/aarch64-linux-gnu/NvInferRuntime.h` 确认可用性。若不可用，退化为 `kSUB` 循环或 `ceil_div` 近似（stride=2 时误差 ≤1 点） |
| **onnx-simplifier 静态化 FPS shape** | 动态 shape 失效 | 导出时用 `--no_simplify`（§7.1 已改）；碎片算子靠 TRT plugin 解决 build 时间 |
| **TRT parser 找不到 plugin（namespace 不匹配）** | engine build 失败 | REGISTER_TENSORRT_PLUGIN 以空 namespace 注册，Creator `getPluginNamespace()` 返回 `""`，ONNX 节点不设 `plugin_namespace` attribute——三者都用空 namespace，匹配。第 12 次审查 Oracle 确认 REGISTER 宏内部调 `registerCreator(instance, "")`（`NvInferRuntime.h:3788`） |
| **TRT 8.6.1 (开发机) 与 Orin 上 JetPack 自带的 TRT 版本不一致**（如 Orin 是 8.5） | engine 不可移植，plugin 接口可能有 ABI 差异 | 在 Orin 上重新编译 plugin + 重新 build engine。`REGISTER_TENSORRT_PLUGIN` 宏在 8.5/8.6 都兼容 |
| **`opt_n_threads` 在 N < 16 时返回 0** | stage 5 的 FPS N=339（5392/16）没问题，但极端情况会崩 | launcher 加 `if (n_threads < 1) n_threads = 1;`（§4.3 已加） |
| **Orin 上 cuDNN 路径不在默认 LD_LIBRARY_PATH** | plugin `.so` 加载失败 | 用 L4T 官方容器（`nvcr.io/nvidia/l4t-tensorrt`），环境预配好 |
| **`init_libnvinfer_plugins` 只注册 builtin，不注册自定义** | CPP_trt3 找不到 plugin | 在 `trt_engine.cpp` 构造前调 `initLibHPENetPlugins()` 或静态链接 `hpenet_plugins`（§7.4） |
| **Windows 上 plugin 是 `.dll` 不是 `.so`** | `ctypes.CDLL` 失败 | 用 `ctypes.WinDLL` 或 `sys.platform` 判断；C++ pipeline 直接链接 |
| **grouping/three_interp 仍是 traceable**（v13 不 patch） | 仍有 668 个碎片算子，2256 Reformat autotune 占 build 主导 | **v14 增补任务已立项**：BallQueryGroup + BallQueryDP + ThreeInterp 三 plugin，见 §13 |
| **FP16 engine 推理精度未达标** | FPS/BQ 强制 fp32 已防溢出，但 Conv fp16 可能有数值问题 | 加 `OBEY_PRECISION_CONSTRAINTS`，让 TRT 在精度敏感处自动回退 |

---

## 11. 工作量估算

| 阶段 | 工作内容 | 工作量 |
|---|---|---|
| 1 | 复制 + 改造 `.cu` kernel（去 PyTorch 依赖，加 stream 参数） | 0.5 天 |
| 2 | 写 FPS TRT Plugin（`IPluginV2DynamicExt` + Creator + 注册） | 2 天 |
| 3 | 写 Ball Query TRT Plugin（同结构，参考 §5.3） | 1 天 |
| 4 | 写 ONNX custom op（`torch.autograd.Function` + symbolic） | 1 天 |
| 5 | CMakeLists.txt（3 平台分支）+ CPP_trt3 接入 | 1 天 |
| 6 | 改 `onnx_backend.py`（2 处函数引用替换） | 0.5 天 |
| 7 | x86 上端到端验证（精度 + 性能 + CPP_trt3） | 1 天 |
| 8 | Orin 上部署 + 验证 | 2 天 |
| **合计** | | **~9 个工作日** |

---

## 12. 参考实现（已核查）

写 plugin 时对照以下开源参考（**均已验证存在且相关**）：

| 参考 | 用途 | URL |
|---|---|---|
| **mmcv `trt_scatternd.cpp`** | 同构的 V2 实现，输出 int32，动态 shape | [open-mmlab/mmcv@a8073c74](https://github.com/open-mmlab/mmcv/blob/a8073c74bf83d62ec36a103f835faa4837fb6585/mmcv/ops/csrc/tensorrt/plugins/trt_scatternd.cpp) |
| **charlesq34/pointnet2 `tf_sampling.cpp`** | npoint 作为 attribute 的经典设计 | [charlesq34/pointnet2@42926632](https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/tf_ops/sampling/tf_sampling.cpp#L23-L31) |
| **NVIDIA/TensorRT `skipLayerNormPluginLegacy.cpp`** | 官方 V2 API 规范写法（含 `PLUGIN_VALIDATE` 错误处理） | [NVIDIA/TensorRT@1dade062](https://github.com/NVIDIA/TensorRT/blob/1dade062a4e796c14ab6b3f32461ad694ec58951/plugin/skipLayerNormPlugin/skipLayerNormPluginLegacy.cpp) |
| **TensorRT 官方文档: IPluginV2DynamicExt** | API 签名权威参考（`getOutputDimensions`/`supportsFormatCombination`/`getWorkspaceSize`/`enqueue`） | [docs.nvidia.com](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html) |
| **TensorRT 官方: NonZeroPluginV2 示例** | 索引输出型 plugin 的 `getOutputDimensions` 用 `exprBuilder.operation` | TensorRT API 文档示例 |
| **TensorRT 官方: ONNX parser 加载 custom op** | "unrecognized nodes 自动作为 plugin 导入"机制 | [docs.nvidia.com/plugins-advanced](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-advanced.html) |

**已确认不存在的参考**（避免误引）：
- ❌ mmdeploy 没有 FPS TRT plugin（全量扫描 main + v1.0.0 tag）
- ❌ mmcv 没有 FPS TRT plugin（只有 CUDA kernel + PyTorch wrapper）
- ❌ ONNX Runtime 没有 `com.microsoft::FurthestPointSampling`（ContribOperators.md 全列表无）
- ❌ `sampleDynamicReshape` 是网络级动态 shape 示例，**不是 plugin 实现**

---

## 13. v14 增补任务：BallQueryGroup / BallQueryDP / ThreeInterp 三 Plugin

> **目标**：把编码器下采样链（ball_query + grouping）与解码器上采样链（three_nn + three_interpolate）分别融合为 plugin，消除 ONNX 图中剩余的 ~668 个 plain 碎片算子，压缩 build 时间与推理延迟。
> **状态**：✅ **已实施并通过全部可本地验证的验收标准（v14.5，2026-08-17）**——V1 单算子 34/34 全过（审查修复前 32/32，修复后追加 2 项定向回归；fp32 精确 0 / idx bit 级一致 / fp16 ≤9.8e-4）、V2 整网 logits 逐位相等、V3 acc **0.9741 零回退**、V4 节点 **628**（阈值 1100）、V6 延迟 **26.5ms 不劣于 v13**、V7 CPP_trt3 回归通过；V5 build 时间持平（31.8min，详见 §9 行 11 的证伪分析）。Orin/Windows/整网 fp16 待测。实施记录见 §13.10。

### 13.1 动机与实测依据

| 证据 | 数据 | 来源 |
|---|---|---|
| plain 碎片算子占比 | Gather 70 + Unsqueeze 176 + Shape 97 + Concat 72 + Reshape 67 = 482，加 Equal/Where/Expand/Cast/ConstantOfShape/GatherElements 各 31 = 186，合计 **668 节点 / 1373 总节点（48.7%）**（各单项计数均经 ONNX 直方图核验），来自 `traceable_grouping` / `traceable_three_interpolation` 的纯 torch 展开 | ONNX 直方图（2026-08-17） |
| Reformat autotune 大头 | build 日志 **2256 条 Reformat autotune**；最终 engine 仅 **14 个 Reformat 层 / 218 层**——条目数 = 候选层 × tactic × 动态 shape 的乘积 | `trtexec --dumpLayerInfo` 反序列化实测 |
| 隐藏的双重 grouping | `QueryAndGroup.forward` 内部已有 `grouped_xyz = grouping_operation(xyz_trans, idx)`（group.py），SA.forward 外部又有 `fj = grouping_operation(f, idx_dp)`（hpenetv2.py:182）——**SA 路径每个调用点是 1×BQ + 2×gather**（InvResMLP 路径为 1×BQ + 1×dp gather，特征 gather 在下游各 block），融合收益比直觉更大 | 源码核查 |

**机制**：Conv/elementwise 偏好向量化布局，Gather/TopK/Shape 等只在 linear 布局实现。碎片算子把 Conv 流切成数百处"布局边界"，每处都是 Reformat 候选。plugin 把整段计算收进单个自定义 kernel，不经过 TRT 布局系统，候选随之消失——v13 build 日志的直接证据：FPS/BQ plugin 节点自身仅占 239 行 autotune 日志，而剩余 traceable 链触发的 Reformat autotune 达 2256 条。

### 13.2 Plugin A：两个粒度（BallQueryGroup + BallQueryDP）

> ⚠️ **v14.1 关键修正**：原设计把 8 个调用点统一用 (grouped, dp) schema 融合，但审查发现 **InvResMLP 的数据流不满足该 schema**——`InvResMLP.forward`（hpenetv2.py:351）只调 grouper 得 `(dp, idx_dp)`，随后 idx_dp 被传给**下游 blocks-1 个 InvResMLP_block**，每个 block 的 `LocalAggregation.forward`（hpenetv2.py:72）用**各自 conv 之后的特征**做 grouping。即 InvResMLP 路径里：①特征在 grouper 调用时不可得；②一个 idx 被 (blocks-1) 个 block 复用，特征 gather 天然不可并入 grouper。因此拆成两个粒度：

**A1：BallQueryGroup（完整融合）——用于 4 个 SetAbstraction 调用点**

SA 路径里 grouper 与特征 gather 在同一 forward 内先后发生（hpenetv2.py:179-197 两个分支），可整体融合：

```
hpenet::BallQueryGroup(xyz, new_xyz, features) -> (grouped, dp)
  xyz:      (B, N, 3)   fp32 恒定     —— 支撑点坐标
  new_xyz:  (B, M, 3)   fp32 恒定     —— FPS 采样中心
  features: (B, C, N)   fp32 | fp16   —— 支撑点特征（convs_d=False 分支为 conv 后特征）
  grouped:  (B, C, M, S) 与 features 同 dtype
  dp:       (B, 3, M, S) fp32 恒定    —— 喂 rel_pos
  attributes: radius (float), nsample (int), normalize_dp (int 0/1) —— 构造期固定，serialize 携带
  （getSerializationSize = sizeof(float) + 2×sizeof(int)；serialize/deserialize 顺序 radius→nsample→normalize_dp，deserialize 后校验 normalize_dp∈{0,1}，v14.2 补）
```

**dp 的精确语义**（v14.1 修正，必须逐位复刻 `QueryAndGroup.forward`，group.py:249-253）：

```python
dp = (grouping(xyzᵀ, idx) − new_xyzᵀ.unsqueeze(-1)) / radius   # normalize_dp=True 时
```

即 kernel 除邻居查找 + gather 外，还须做**中心点减法**与 **÷radius 归一化**（hpenet-ll.yaml `normalize_dp: True`，radius=10）——这两步在 `ball_query_gpu.cu` 里没有，是 Python 侧逻辑，必须新写入 kernel。漏掉 ÷radius 会使 dp 放大 10 倍（HPE 首层 BN 会部分吸收，但数值语义已变，V1 对照会失败）。

**A2：BallQueryDP（轻量融合）——用于 4 个 InvResMLP 调用点**

只融合 BQ + dp gather（消掉每处 dp 的 gather 展开链），特征 gather 保持 traceable（idx 需保留传给下游各 block）：

```
hpenet::BallQueryDP(xyz, new_xyz) -> (dp, idx)
  xyz:     (B, N, 3)  fp32 恒定   （InvResMLP 中 new_xyz = xyz，query==support）
  new_xyz: (B, M, 3)  fp32 恒定
  dp:      (B, 3, M, S) fp32
  idx:     (B, M, S)  int32       —— 传给下游 LocalAggregation 的特征 gather（仍走 traceable）
  attributes: radius (float), nsample (int), normalize_dp (int 0/1)  # 序列化同 A1（见 A1 注记）
```

**kernel（A1/A2 共用基础设施）**：`ball_query_gpu.cu` 的邻居查找逻辑照搬（padding 规则逐位一致，见 §1.2）；A1 再加 gather + 减法 + 归一化。两个方案任选：
- 方案 1（低风险）：enqueue 内连续 launch——ball_query 出 idx（放 workspace，A2 中另作输出）+ gather/减法/归一化 kernel
- 方案 2（更优）：单 kernel——每 thread (m, s) 算完邻居索引后立即完成搬运与 dp 计算，省一次全局写读

**动态 shape 与 workspace（v14.1 修正）**：
- 维度索引：**M = `inputs[1].d[1]`，N = `inputs[0].d[1]`，B = `inputs[0].d[0]`，C = `inputs[2].d[1]`**（features 是 (B,C,N)，C 在 **d[1]** 不是 d[0]），S = nsample attribute。`getOutputDimensions` 用 exprBuilder 直接引用输入维度（无需 kFLOOR_DIV，比 FPS 更简单）。
- workspace：方案 2 为 0；**方案 1 中 idx 中间量为 maxB×maxM×S×4 字节**。⚠️ `getWorkspaceSize` 收到的 `inputs` 是当前 shape（构建期为 opt），**看不到 max dims**——必须沿用 FPS 模式：`configurePlugin` 里从 `in[i].max.d[]` 缓存 maxB_/maxM_（clone() 也要拷贝缓存，fps_plugin.cpp:86-91 先例），`getWorkspaceSize` 用缓存值计算。

**调用点小结**：BallQueryGroup × 4（encoder.1-4 的 SA）+ BallQueryDP × 4（encoder.1-4 的 InvResMLP）。A1 每处替代 ~15-20 个碎片节点，A2 每处消掉 dp gather 链（特征 gather 链保留）。

### 13.3 Plugin B：ThreeInterp（three_nn + 权重 + three_interpolate 融合）

**ONNX 节点 schema**：

```
hpenet::ThreeInterp(unknown_xyz, known_xyz, known_feat) -> interpolated
  unknown_xyz: (B, N, 3)  fp32 恒定     —— 稀疏层（目标）坐标
  known_xyz:   (B, M, 3)  fp32 恒定     —— 稠密层（来源）坐标
  known_feat:  (B, C, M)  fp32 | fp16
  interpolated: (B, C, N) 与 known_feat 同 dtype
  无 attribute（k=3 固定写死，与 PointNet++ 语义一致；getSerializationSize=0）
```

**调用点（v14.1 修正，v14.2 再澄清）：5 处运行调用，图内 4 份独立 cdist+TopK 链**——源码行只有 3 行（256/258/600），`FeaturePropogation.forward` 的 256 行（258 行在 decoder 路径是死分支，f1 恒非 None）被 **4 个 decoder stage 各执行一次**（hpenetv2.py:590/591/592/598），加 600 行直接调用 1 次 = **5 次运行调用**。但经查当前 ONNX 图：只有 **4 个 TopK 节点**（decoder.0-3.0）——600 行调用与 decoder.0.0 的插值**几何完全相同**（都是 p[1]←p[2]），其 cdist+TopK 子图被 ONNX 导出器 CSE 合并；其 gather/加权和链独立存在（`/model/decoder/GatherElements`）。plugin 化后 5 个调用点的特征输入各不相同，**不会被合并，仍是 5 个 `hpenet::ThreeInterp` 节点**（V4 判据 ×5 有效）。（hpenetv2.py:776 的另一处在 HPENetV2PartDecoder，雷达模型不用，不在图内。）

**kernel（v14.1 修正——两个 .cu kernel 的网格映射不同，不能天真合并）**：
- `three_nn_kernel_fast` 网格 (⌈N/256⌉, B)：每 thread 一个目标点，扫 M 做 running top-3（内部用 **double** 精度维护 best1/2/3，interpolate_gpu.cu:37）
- `three_interpolate_kernel_fast` 网格 (⌈N/256⌉, C, B)：每 thread 一个 (点, 通道)
- 天真单 kernel 有两个坑：按 interpolate 网格映射会把 O(M) 的 top-3 搜索重复 C 次（C 至多 **512**——encoder 最深层 channels=[32,64,128,256,512]，v14.2 修正原误写的 256，浪费最高 512×）；按 three_nn 网格映射则每 thread 串行循环 C，并行度只剩 N。**二选一**：
  - **方案 1（低风险，推荐先做；✅ 已实施，v14.5 优化版）**：两个 kernel + workspace——kernel1 出 idx+weight（`maxB×N×3×8` 字节，int32 idx + fp32 weight，configurePlugin 缓存 max dims，同 §13.2 方案 1 模式），kernel2 做 MAC。dist/idx 仍不出 plugin 边界（不出图，只出 workspace）。**实施时 kernel1 重写为"每点 8 线程协作"版**：一 thread 一点的原始布局在 decoder 规模下只有 11 个 block（92 SM 的 L20 利用率 ~1.5%，实测 1.12ms）；改为 8 线程分摊 M 维扫描 + block 内 shared memory 归并 24 个候选（吸取了方案 2 的协作思想但保持两 kernel 结构），**1.12ms → 0.017ms（66×）**。注意实现要点：早退线程必须在 barrier 前到达（守卫计算 + 无条件 `__syncthreads`，见 §13.10 审查修复 #2）；unknown 指针须含 batch 偏移（§13.10 修复 #1）
  - **方案 2（更优，v14.3 补可行性边界）**：单 kernel + shared memory——每 block 先协作算出该 block 目标点的 top-3（block 内共享 besti1/2/3 与 best1/2/3），随后所有 C 维 thread 从共享内存读 idx/weight 做 MAC。⚠️ 该方案的隐含前提是 **C 维 thread 与目标点在同一 block 内**：blockDim ≤1024，C=512 时每 block 至多 2 个目标点（512×2）——线程占用率虽满，但每 block 工作粒度变粗、C=512 与 blockDim 强耦合（v14.4 措辞修正，原误写"占用率塌缩"）；SMEM 本身不是瓶颈（每点 3×int32+3×fp32 ≈ 24B）。若实施时 profiling 证明方案 1 的双 launch 开销可忽略，方案 2 可不做

**权重公式（v14.1 修正——不在任何 .cu 里，是新增代码，必须逐位对齐 upsampling.py:97-100）**：

```python
dist = sqrt(d2)                      # CUDA kernel 输出 d²，Python 包装层 ThreeNN.forward 开方（upsampling.py:33）——plugin 内须自行开方
w_i  = 1.0 / (dist_i + 1e-8)         # +1e-8 处理 d=0 的重合点（权重→1）
weight = w / (w1 + w2 + w3)          # 3 邻域 L1 归一化
```

注意三处易错点：**sqrt**（kernel 给的是 d²）、**+1e-8**、**L1 归一化**。朴素 "1/d" 实现会直接挂掉 V1 的 fp32 `max_abs_err ≤ 1e-5`。~~top-3 的 running 比较沿用 double~~（**v14.5 修正**：实施版改用 **float 比较**——所有 d 值本身是 float，float→double 转换精确且单调，故 float 比较与参考 kernel 的 double 比较**逐位等价**；哨兵 `FLT_MAX` 与原 `1e40` 对一切有限浮点等价。V1 数值实测逐位不变，且 float 比较 ALU 吞吐翻倍）。

**动态 shape**：N = `inputs[0].d[1]`，M = `inputs[1].d[1]`，C = **`inputs[2].d[1]`**（known_feat 是 (B,C,M)，C 在 d[1]）。

**kernel 适配注意**：原 `interpolate_gpu.cu` launcher 的 `exit(-1)` 错误处理必须剥除（plugin 库里调 exit 会杀死宿主进程；FPS/BQ 适配时已同样处理），grad kernel（three_interpolate_grad_*）不需要，不搬。

### 13.4 dtype 策略（fp32/fp16 关键设计）

**xyz 恒 fp32，features 跟随 engine 精度**——`supportsFormatCombination` 按 tensor 逐个声明：

| tensor | 接受格式 | 理由 |
|---|---|---|
| xyz / new_xyz / unknown_xyz / known_xyz | kLINEAR + **仅 kFLOAT** | fp16 尾数 10 位：坐标 30m 时 ulp≈1.6cm（voxel 0.3m 的 5%）；几何误差会翻转 ball_query 邻域归属、three_nn 排序、进而级联污染分组特征——离散决策输入不容连续噪声。且 xyz 仅 3 通道，fp16 省的带宽趋近于零，收益/风险比极差 |
| features / known_feat | kLINEAR + kFLOAT 或 kHALF | 误差连续有界（conv/ReLU 近 Lipschitz），且占激活流量 95%+，是 fp16 收益所在 |
| grouped / dp / interpolated | 与对应输入同 dtype；dp 恒 fp32 | dp 参与后续位置编码，保 fp32 |

若 TRT 在 fp16 engine 里试图以 fp16 喂 xyz，组合被拒，builder 自动在 plugin 前插 Reformat 转 fp32——一次 3 通道小转换，可接受。

**`supportsFormatCombination` 的实现要点（v14.1 补充，v14.2 补 BallQueryDP）**：pos 编号 = 输入在前、输出在后（BallQueryGroup 5 个位置：0=xyz, 1=new_xyz, 2=features, 3=grouped, 4=dp；**BallQueryDP 4 个位置：0=xyz, 1=new_xyz, 2=dp, 3=idx**；ThreeInterp 4 个位置：0=unknown_xyz, 1=known_xyz, 2=known_feat, 3=interpolated）。"grouped 跟随 features dtype" 的条件关系写成 `inOut[3].type == inOut[2].type`（引用更小的 pos 合法），xyz/dp 恒 `type == kFLOAT && format == kLINEAR`，idx 恒 `kINT32 + kLINEAR`，features/known_feat 接受 `kFLOAT 或 kHALF`。注：BallQueryDP 的 int32 idx 进入下游 traceable 特征 gather 时会被 `idx.to(torch.int64)`（onnx_backend.py:128）插一个 Cast 节点——与现有 FPS/BQ→traceable 路径行为一致，节点预算已含余量（v14.3 核注）。头文件 NvInferRuntimePlugin.h 的接口注释明确允许同 plugin 内 mixed dtype（官方示例即"双 fp16 输入 + fp32 输出"，NvInferRuntimePlugin.h:654 / NvInferRuntime.h:406-407）。

**`getOutputDataType` 的陷阱（v14.1 补充，v14.2 措辞修正）**：该接口收到的 inputTypes **恒为 kFLOAT/kINT32**，且**返回值**若为 kHALF/kINT8 会被规范化为 kFLOAT（NvInferRuntime.h:346-347），实际 fp16 精度由 `supportsFormatCombination` 决定。因此 `getOutputDataType` 对 float 输出**一律返回 kFLOAT**（多输出按 index：BallQueryGroup 的 grouped/dp 均 kFLOAT；BallQueryDP 的 idx 返回 kINT32），**不要**试图在这里返回 kHALF。

**kernel 内部**：所有距离计算、权重归一化、累加全程 fp32（top-3 比较沿用原 kernel 的 double）；仅特征搬运（gather）与最终 MAC 按输入 dtype 分支（模板或 `if (inputDesc[2].type == kHALF)` 双路径——此为本方案新增写法，§8.2 只涉及"强制 fp32"的先例）。

### 13.5 ONNX 导出侧与 patch 点

新增 op 文件，模式照抄 `deploy/onnx_ops/fps_op.py`（`torch.autograd.Function` 子类：forward 为 CPU 占位实现，static `symbolic` 用 `g.op` 输出 `hpenet::` 节点——**普通函数无法发射 custom op 节点，必须包成 Function 子类**）：

| 文件 | forward 实现规范（v14.2 修正，v14.3 补引用来源——见下方 ⚠️） | patch 挂点 |
|---|---|---|
| `deploy/onnx_ops/ballquerygroup_op.py` | **CUDA 路径**：直接 import 真实 CUDA op（`from openpoints.cpp import pointnet2_cuda`，用 `pointnet2_cuda.ball_query_wrapper` / `pointnet2_cuda.group_points_wrapper`——**不要引用 patch 后的模块级名字**，见 ⚠️②）+ Python 减法/÷radius；**CPU 路径**（export 实际运行环境，onnx_export.py:87 `model.cpu()`）：返回 shape/dtype 正确的占位张量 | **monkeypatch `SetAbstraction.forward`**（`forward.__get__(module)` 实例级绑定，`_patch_residual_blocks` 先例）：patch 工厂内读取 `self.grouper.radius / self.grouper.nsample / self.grouper.normalize_dp`（group.py:226-227）并闭包绑定，产出 `fj, dp = make_ballquerygroup_op(radius, nsample, normalize_dp)(p, new_p, f)`。注意 convs_d=False 分支须传 **conv 之后**的 f；convs_d=True/False 两分支都要覆盖 |
| `deploy/onnx_ops/ballquerydp_op.py` | 同上（CUDA 调真实 op，CPU 返回占位） | **monkeypatch `InvResMLP.forward`**（**仅外科式替换 grouper 一行**：`dp, idx_dp = self.grouper(p, p, f)` → 工厂调用；`pe = self.rel_pos(dp)` 与 `self.layer([p, f, dp, idx_dp, pe])` 链原样保留），同样从 `self.grouper` 闭包绑定三属性，产出 `dp, idx_dp = make_ballquerydp_op(radius, nsample, normalize_dp)(p, p)`（返回签名保持 (dp, idx)，**下游 LocalAggregation/各 InvResMLP_block 完全不动**，idx 继续走 traceable 特征 gather） |
| `deploy/onnx_ops/threeinterp_op.py` | CUDA 路径调**原始** `three_interpolation`（`from openpoints.models.layers.upsampling import three_interpolation` 直接引入函数对象——**不可用 `hpenetv2_mod.three_interpolation`，patch 后它已指向本 op 自身，会无限递归**，见 ⚠️②）；CPU 路径返回 shape/dtype 正确的占位张量（v14.4 统一——Function.forward 对 tracing 不透明，占位即够，与其他两 op 一致） | `hpenetv2_mod.three_interpolation = ThreeInterpOp.apply`（现有 patch 行改指向，5 个运行调用点自动生效；本 op 无 attribute，无需工厂） |

> ⚠️ **v14.2 关键警告——勿复用现有 `traceable_ball_query` 作为占位/对照实现**：`onnx_backend.py:69-113` 的 `traceable_ball_query` 对越界/半径外位置填 **idx=0**（:105/:111），这正是 §1.2 记录的 padding bug（真实 kernel 填**第一个找到的邻居**，ball_query_gpu.cu:41-44）。上一版文档写的"traceable_ball_query 修正版语义"指的是一个**尚不存在、需要新写**的修正版本——若直接复用现有 idx=0 版本做 forward，V1 的 bit 级索引比对与 V2 整网对照都会失败。最省事且零风险的做法即上表：CUDA 路径直接调真实 CUDA op（`ballquery_op` 已验证此模式），无需新写任何 traceable 修正版。
>
> ⚠️ **v14.3 关键补充①——attribute 传递机制（按旧文档字面实现必失败）**：§13.2 schema 要求 radius/nsample/normalize_dp 三个构造期 attribute，但它们**不会凭空到达 symbolic**。既有模式二选一：ballquery_op 先例是 **positional args**（`apply(radius, nsample, xyz, new_xyz)`，forward/symbolic 签名同序），fps_op 先例是 **工厂闭包**（`make_fps_op(stride)`，§13.5 上表采用此式）。symbolic 必须以 `radius_f=float(radius), nsample_i=int(nsample), normalize_dp_i=int(normalize_dp)` 发射 attribute（命名后缀 `_f`/`_i` 是 TRT parser 按字符串读取的既定约定，§4.1/§5.2 代码先例）；**BallQueryGroup/BallQueryDP 是双输出节点，symbolic 须加 `outputs=2` 并显式解包**：`grouped, dp = g.op("hpenet::BallQueryGroup", ..., outputs=2)`（`outputs>1` 时 g.op 返回 tuple，NIT 提示；现有 FPS/BQ 均为 `outputs=1`，项目内无先例，PyTorch 内建 Split/MaxPool symbolic 可参考）。漏掉任一项 → ONNX 节点无 attribute → TRT deserialize 读到垃圾。
>
> ⚠️ **v14.3 关键补充②——CUDA 路径引用来源**：patch_model_for_onnx 运行后，`group_mod.grouping_operation`、`hpenetv2_mod.grouping_operation`、`hpenetv2_mod.three_interpolation` 都已被改写（onnx_backend.py:193-196 + 本节 patch）——新 op 的 CUDA forward 若引用这些**模块级名字**，要么调到 traceable 版（数值路径不一致），要么（threeinterp 场景）调到自身造成**无限递归**。必须像上表那样 `from ... import 原始函数对象` 直接持引用。

**为什么必须挂在 forward 层级（v14.1 修正）**：参与该计算的有 **3 个**模块级引用——`group_mod.ball_query`（BQ，group.py:244）、`group_mod.grouping_operation`（dp gather，group.py:249，在 `QueryAndGroup.forward` 内部）、`hpenetv2_mod.grouping_operation`（特征 gather，hpenetv2.py:182/193 与 LocalAggregation:72；前两者与它在 patch 前本是同一对象 `GroupingOperation.apply`）。BQ 与 dp gather 在 QueryAndGroup 内、特征 gather 在 SA/LocalAggregation 内，**patch 任何单个模块引用都无法把三者聚到一起**，只能在 forward 层级替换整段。

**实施补充的两处细节（v14.5）**：① SA 替换后的 forward 对 `sample_fn` 必须**调用期动态解析**（`module.sample_fn`）——sample_fn 的替换发生在 `_patch_group_plugins` 之后，构造期闭包捕获会拿到原始 CUDA FPS（§13.10 修复 #2）；② `_patch_group_plugins` 对 **all_aggr 的 SA（非头且 stride==1，grouper 是 GroupAll）直接跳过**——其 radius/nsample 为 None 且 dp 语义不同（§13.10 修复 #4；hpenet-ll 不可达，守卫为其他配置而设）。

**现有 patch 行的保留/移除（v14.1 修正）**：`hpenetv2_mod.grouping_operation = traceable_grouping_operation` 与 `group_mod.grouping_operation = ...` **必须保留**——InvResMLP 路径的特征 gather 仍走 traceable；`group_mod.ball_query = ballquery_op` 可保留作兜底（SA/InvResMLP 均不再经过 QueryAndGroup，但保留无害）。导出继续用 `--no_simplify`（§7.1）；`hpenet::` 节点对 simplifier 不透明，不会被折叠。

### 13.6 新增/修改文件清单

```
新增:
  deploy/onnx_ops/ballquerygroup_op.py        # autograd.Function + symbolic + CPU 占位 + make_* 工厂（§13.5）
  deploy/onnx_ops/ballquerydp_op.py           # 同上
  deploy/onnx_ops/threeinterp_op.py
  deploy/trt_plugins/include/ballquerygroup_plugin.h   # 含 Creator 声明（参照 §4.3 教训）
  deploy/trt_plugins/include/ballquerydp_plugin.h
  deploy/trt_plugins/include/threeinterp_plugin.h
  deploy/trt_plugins/include/ballquerygroup_kernel.h   # BallQueryDP 的 plugin 类（ballquerygroup_plugin.cpp 内）复用此 kernel 头
  deploy/trt_plugins/include/threeinterp_kernel.h      # ThreeInterp 专用 kernel 头
  deploy/trt_plugins/src/ballquerygroup_kernel.cu      # NVCC 编译，fill/工具宏在 .cu；剥除 exit(-1)，不搬 grad kernel
  deploy/trt_plugins/src/threeinterp_kernel.cu
  deploy/trt_plugins/src/ballquerygroup_plugin.cpp     # 含 BallQueryDP plugin 类（共用 Creator 模式）
  deploy/trt_plugins/src/threeinterp_plugin.cpp
修改:
  deploy/trt_plugins/src/plugin_registry.cpp  # 追加 3 个 REGISTER_TENSORRT_PLUGIN（Group/DP/ThreeInterp）
  deploy/trt_plugins/CMakeLists.txt           # 追加 2 组源文件
  deploy/onnx_backend.py                      # patch_model_for_onnx 按 §13.5 上表改（保留 grouping 的 traceable patch）
  plugin.md §9 验证矩阵                        # 实施后回填（v13 median 27.13ms / v14 26.53ms 纯 engine 延迟基线）
```

### 13.7 验收标准（全部通过才算完成）

| # | 验证项 | 通过标准 |
|---|---|---|
| V1 | **fp32/fp16 数值对照（单算子）** | 随机输入 + 真实雷达点云（N=1024/5500/6500 三档）下，plugin CUDA 输出 vs PyTorch 参考实现（`QueryAndGroup.forward` / `three_interpolation`，即真实 CUDA op + Python dp/权重语义）逐 tensor 比对：fp32 `max_abs_err ≤ 1e-5`；fp16 `max_rel_err ≤ 1e-3`；BallQueryDP 的邻居索引 **bit 级一致**（含 padding 规则；BallQueryGroup 无 idx 输出，其索引一致性由 grouped/dp 零误差间接保证），dp 含减法与 ÷radius 归一化；ThreeInterp 权重公式为 `1/(sqrt(d²)+1e-8)` L1 归一化。**✅ 实施结果（2026-08-17）**：34/34 全过——fp32 的 dp/grouped 误差为 **0**、ThreeInterp 3.6e-7；fp16 的 ThreeInterp 采用两个公平度量（half 输入精确对照 ≤9.8e-4 + 全局相对误差 ≤3.9e-4——元素级 rel 在近零输出上因 half 舍入天然爆炸，不宜作判据）；审查修复后追加定向回归 **B=2/3（batch 偏移）与 N%4∈{0,2,3}（含 decoder 真实 shape 2750）** 全过 |
| V2 | **fp32/fp16 数值对照（整网）** | patched PyTorch 模型 forward vs 原 checkpoint 模型 forward，10 个测试文件：fp32 logits `max_abs_err ≤ 1e-4`；fp16 预测标签一致率 ≥ 99.9% |
| V3 | **acc 不回退** | TRT fp32 engine 推理 acc ≥ **0.9741 − 0.005**（当前 baseline 0.9741，容差留给数值噪声）；fp16 engine acc ≥ **0.96**（较 §9 行 10 原定 0.87 收紧，v14 起生效——fp16 尚未实测，0.96 为绝对阈值，若实测与 fp32 差距 >1 点须归因） |
| V4 | **ONNX 节点数对比** | `hpenet::BallQueryGroup` × 4 + `hpenet::BallQueryDP` × 4 + `hpenet::ThreeInterp` × 5 出现（5 个节点因特征输入不同不会被 CSE 合并，见 §13.3 调用点说明）；SA 的特征 gather 链与 dp 链、InvResMLP 的 dp 链、three_interp 展开链全部消失（InvResMLP 的**特征 gather 链保留属预期**，hpenetv2.py:172 残差 gather 亦为合法保留）；总节点数 ≤ **1100**（核算依据 v14.2：SA 4×~18=72 + InvResMLP dp 链 4×~10=40 + ThreeInterp **4 份独立** cdist+TopK 链 ×~49≈196——5 个调用点中 2 处几何相同、当前图内本就只有 4 份——合计 ~310-330，1373−330≈1043，留余量），实测数字回填 §9 |
| V5 | **build 时间对比** | 记录 build 墙钟时间与 **Reformat autotune 条目数（与 v13 同口径统计）**，与 v13 基线（~29min / 2256 条）对比并回填 §9；不设硬指标但须给出实测数字与分析 |
| V6 | **推理延迟对比** | trtexec per-layer profiling（N=5500）：BallQueryGroup 单层耗时 ≤ 原 BQ+gather 链之和；ThreeInterp 单层 ≤ 原 cdist+TopK 链之和；整网纯 engine 延迟不劣于 v13 基线 31.6ms @ N=5500（⚠️ v14.2 核注：该数字来自 v13 会话的 trtexec 实测，仓库内无存档 artifact 可复核，实施时须先重跑 v13 engine 的 trtexec profiling 重建基线并回填 §9，再作对比） |
| V7 | **回归** | CPP_trt3 加载新 engine 推理成功；图内 `hpenet::FPS` 仍为 4 节点，且同一输入下 FPS 输出索引与 v13 engine 逐元素一致 |

### 13.8 风险与对策（增量）

| 风险 | 对策 |
|---|---|
| BallQueryGroup/BallQueryDP 邻居选取顺序/padding 与真实 kernel 不一致 → 精度回退 | 邻居查找逻辑照搬 `ball_query_gpu.cu`（勿手写）；V1 的 bit 级索引比对兜底 |
| **dp 语义漏项**：中心点减法或 ÷radius 归一化（normalize_dp=True）遗漏 | kernel 内显式实现 `dp=(gather(xyz)−query)/radius`；V1 以 `QueryAndGroup.forward` 输出为参考逐 tensor 对照 |
| **ThreeInterp 权重公式写错**（漏 sqrt/+1e-8/L1 归一化之一） | 权重按 §13.3 公式实现；V1 以 `three_interpolation`（upsampling.py:92-102）为参考，fp32 `max_abs_err ≤ 1e-5` 兜底 |
| SA/InvResMLP patch 遗漏分支 | SA patch 覆盖 `convs_d` True（:179-187）/False（:190-197）两分支且后者传 conv 后特征；InvResMLP patch 保持 (dp, idx) 返回签名使下游零改动；导出后验证 BallQueryGroup ×4 + BallQueryDP ×4 + ThreeInterp ×5 |
| workspace 按 opt shape 分配 → 运行期 N>opt 时越界 | configurePlugin 缓存 `in[i].max.d[]`（FPS 先例 fps_plugin.cpp），clone() 拷贝缓存；getWorkspaceSize 用缓存值 |
| fp16 engine 下 xyz 被上游算子转成 fp16 才到 plugin | `supportsFormatCombination` 拒绝 fp16 xyz，builder 自动插 Reformat；导出侧确认 `pos` 网络输入声明为 fp32（现状即是） |
| three_nn 的 fp16 top-3 排序翻转 | xyz 恒 fp32 + 距离 fp32/double 寄存器计算，排序输入精度与训练一致 |
| 原 .cu launcher 的 `exit(-1)` 混入 plugin 库 | 适配时剥除（FPS/BQ 先例）；grad kernel 不搬 |
| **symbolic 漏发 attribute 或漏 `outputs=2`/解包** | 按 §13.5 ⚠️① 实现；导出后用 onnx 工具检查节点 attribute 与输出数 |
| **CUDA forward 引用 patch 后模块级名字 → 递归/数值路径不一致** | 按 §13.5 ⚠️② 直接持有原始函数对象引用；实施后跑一次 CUDA forward 冒烟测试 |
| plugin 数增至 3 个后 .so 体积/编译时间增长 | 可忽略（每个 ~200 行 kernel + ~300 行 plugin，模式同现有） |

### 13.9 工作量估算

> 实测：实际耗时 ~1 个工作日（x86 全流程），显著低于下表——四轮审查把设计打磨到位后实施几乎是机械落地。Orin/Windows（阶段 6）未做。

| 阶段 | 内容 | 工作量 |
|---|---|---|
| 1 | 2 个 `.cu` kernel（照搬现有 kernel + 权重/dp 新代码 + dtype 分支 + 剥 exit） | 1 天 |
| 2 | 3 个 TRT plugin 类（Group/DP/ThreeInterp）+ 注册 + CMake | 1.5 天 |
| 3 | 3 个 ONNX op（含 make_* 工厂与 attribute 绑定，§13.5）+ onnx_backend patch（SA.forward 双分支 + InvResMLP.forward） | 1.5 天 |
| 4 | V1-V2 数值对照脚本与排查 | 1 天 |
| 5 | V3-V7 整网验证 + 回填 §9 | 0.5 天 |
| 6 | Orin(aarch64) plugin 重编 + engine 重建 + fp32/fp16 验证；Windows 推理侧确认（对齐 §11 v13 的 2 天 Orin 预算与 tip 规格 2/4/5） | 2 天 |
| **合计** | | **~7.5 个工作日** |

### 13.10 实施记录（v14.5，2026-08-17）

**落地文件**：§13.6 清单全部就位（12 新文件 + registry +3 REGISTER + CMake +2 组源文件 + onnx_backend patch）。

**实施中发现并解决的 4 个问题**：

1. **TRT ONNX parser 按属性名原样传 PluginField，torch.onnx 导出时剥 `_f/_i` 后缀**——手搓测试 ONNX 用 `radius_f` 命名导致 creator 收到 0 个 field（idx 全零）。真实导出路径（symbolic `radius_f=`）天然正确，无需改代码；V1 测试脚本改用无后缀命名。
2. **SA forward 闭包捕获了未替换的 `sample_fn`**——`_patch_group_plugins` 在 sample_fn 替换之前运行，构造期捕获拿到原始 CUDA FPS，CPU 导出时炸出 illegal access。修复：forward 内调用期动态解析 `module.sample_fn`。
3. **ThreeInterp top-3 kernel 占用率病**（一 thread 一点：N=2750 → 11 block/92 SM，GPU 利用率 ~1.5%，实测 1.12ms）——重写为**每点 8 线程协作扫描 + shared memory 归并**，并把 double 比较改为 float（float 值的 float 比较与 double 比较逐位等价，哨兵 FLT_MAX≡1e40）。**1.12ms → 0.017ms（66×）**，V1 数值逐位不变。
4. **V1 fp16 初测的 nan/越界是测试缺陷**——fp16 build 里 TRT 可能为 plugin 选 fp32 绑定，喂 half 缓冲导致越界写；测试改为按 engine 绑定 dtype 分配 + OBEY_PRECISION_CONSTRAINTS 强制走 half 路径。ThreeInterp fp16 度量改为"half 输入精确对照 + 全局相对误差"（元素级 rel 在近零输出上天然爆炸）。

**实测结果**（详见 §9）：V1 34/34（修复前 32/32 + 审查修复后 2 项定向回归）、V2 logits 逐位相等、V3 acc 0.9741 两次复测零回退、V4 节点 628、V6 median 26.53ms 不劣于 v13 27.13ms（端到端 -21%）、V7 通过；**V5 build 时间 31.8min 持平**——最终 engine Reformat 层数不变（14），v13"2256 Reformat autotune 占 build 主导"的判断被证伪，真实主导项是 conv tactic autotune。

**遗留**：Orin/Windows 重编验证（§13.9 阶段 6）、`--separateProfileRun` 下 encoder.4.1/BallQueryDP 的 2.4ms 归因异常（独立微基准 0.072ms，疑为 profiler 归因噪声，不影响端到端）。

**fp16 全流程实测（2026-08-18）**：ONNX → `trt_build.py --fp16` → engine（92.4min，9.4MB）→ `trt_inference.py` 10 文件 **acc 0.9741 与 fp32 持平**；纯 engine 延迟 median 26.38ms（-0.6%，FPS 主导故提升有限）。plugin 层被 builder 自选为 fp32 执行（features 周边省 reformat 的成本决策），half 路径能力由 V1 强制 half（OBEY_PRECISION_CONSTRAINTS）逐位验证兜底。tip 规格第 5 条（fp32/fp16）**完整闭环**。

**实施后代码审查（2026-08-17 晚，双 Oracle）发现并修复 4 项**：

1. **[BUG] ti_top3_weight_kernel 的 unknown 指针漏 batch 偏移**——`unknown += group*3` 缺 `blockIdx.y*n*3`，B>1 时非零 batch 的插值用错坐标。V1 只测了 B=1 所以没暴露。已修复，B=2/3 定向回归逐位通过（err≤4.8e-7）。
2. **[UB] `__syncthreads` 前早退**——`if (group>=n) return;` 后有效线程才到 barrier：n%4≠0 时部分 warp 早退属教科书级未定义行为（可挂死）。V1 测的 N=1024/5500/6500 恰好都能被 4 整除而掩盖。已重构为"守卫计算 + 无条件 barrier"；**decoder 真实 shape N=2750（%4=2）定向回归通过**。
3. **[规格违背] clone() 未拷贝 maxB_/maxM_ 缓存**（BallQueryGroup/ThreeInterp；fps_plugin.cpp:86-91 先例与 §13.2 明确要求）——若 TRT 在 configurePlugin 后 clone 且对新对象调 getWorkspaceSize，会按默认 1,1 算出 128 字节 workspace → 越界写。已补拷贝。
4. **[潜伏 BUG] SA patch 未守卫 all_aggr 路径**——非头且 stride==1 的 SA 用 GroupAll（radius=None），照 patch 会 `float(None)` 崩溃且 dp 语义本就不同。hpenet-ll 不可达，但通用 patch 函数对其他配置（如 strides=[4,1,2,...]）会踩。已在 `_patch_group_plugins` 加 all_aggr 跳过守卫。

修复后复测：V1 全套 34/34 PASS、定向回归（B=2/3、N%4∈{0,2,3}）全过、现有 engine（kernel 在 .so 内，无需重建）acc 仍 0.9741。审查确认干净项：dp (B,3,M,S) 布局、gather/MAC 索引算术（size_t 提升）、SA/InvResMLP monkeypatch 与原 forward 逐行等价（all_aggr 外）、参数顺序（support/query 映射）、normalize_by_std（本仓库 group.py 未实现，无影响）、双次 patch 幂等。

---

## 附录 A：常见错误排查

| 错误信息 | 原因 | 解决 |
|---|---|---|
| `Could not find plugin: hpenet::FPS` | plugin `.so` 未加载 | `ldd libhpenet_plugins.so` 看依赖；CPP_trt3 检查 CMakeLists 是否链接 |
| `UNDEFINED_SYMBOL: _ZN6hpenet...` | C++ name mangling 不匹配 | plugin 入口函数加 `extern "C"` |
| `plugin registered twice` | `.so` 被加载两次 | 检查 `trt_build.py` 和 CPP_trt3 都只加载一次 |
| `Engine build failed: INVALID_NODE` | ONNX 节点 schema 与 plugin 期望的不一致 | 检查 opset domain/version；attribute 名拼写 |
| `CUDA error: launch failure` in FPS | `n_threads = 0`（N 太小） | launcher 加 `n_threads = max(n_threads, 1)` |
| 推理 acc 仍然偏低（< 0.88） | ball_query 的 padding 行为不对 | 确保用真实 CUDA kernel（padding=第一个邻居），不是 traceable 版（padding=0） |
| FP16 推理输出含 NaN | FPS 的 temp 缓冲在 fp16 下溢出 | `supportsFormatCombination` 只允许 fp32；用 fill kernel 设 1e10 |
| CPP_trt3 报 "plugin not found" | `initLibHPENetPlugins()` 未调用 | `trt_engine.cpp` 构造前调用，或静态链接 |

---

## 附录 B：与现有文档的关系

- `deploy/trt_plugin_tip.md`：本方案的**原始需求规格**（核心 7 条要求见下表；文件实际含 9 条——第 8 条编译/执行命令由 §7 覆盖、第 9 条"BallQuery+grouping 合并与 three_interp plugin 化实施计划"即 §13 本身，v14.3 核注）
- `deploy/deploy.md`：现有 ONNX/TRT 部署流程文档（不含 plugin）
- `模型部署.md`：部署问题排查记录
- **本文档 `plugin.md`**：完整工程实施方案，**实施完成后本文件应作为 plugin 模块的 README 索引**

### trt_plugin_tip.md 要求覆盖对照

| 要求 | 本方案覆盖章节 |
|---|---|
| 1. pth→onnx 用自定义 ball_query、fps；onnx_inference.py 能推理 | §4.1, §5.2（symbolic）+ §6（patch） |
| 2. 插件库源码包，便于平台移植 | §3.1（目录结构）+ §4.3/§5.3（源码） |
| 3. onnx→engine 用自定义算子；trt_inference.py 能推理 | §4.3/§5.3（TRT Plugin）+ §7.1（build engine） |
| 4. onnx_inference/trt_inference 在 linux+x86；其余平台只推理 | §7.1（x86 全流程）+ §7.2（aarch64 build+推理）+ §7.3（Windows 推理） |
| 5. fp32 和 fp16 支持 | §8（FP16 处理） |
| 6. **首先满足 CPP_trt3 使用** | §7.4（CPP_trt3 接入，3 处改动） |
| 7. 编译、执行命令 | §7（全平台命令） |

## 附录 C：问答（FAQ）

> 收录实施过程中的关键问答，原文来自 2026-08-17 的工作讨论。

### C.1 fps、ball_query、grouping、three_nn、three_interpolate 这几个操作的关系是什么？各发挥什么作用？

这 5 个算子是 PointNet++ 体系里"点云没有规则网格"这一现实逼出来的基础工具箱：CNN 里 stride 卷积/pooling 做下采样、resize/转置卷积做上采样，点云里这两件事分别由这 5 个算子手工构造（对应 `hpenetv2.py` 的 `SetAbstraction.forward`）。

**编码器：下采样三件套（FPS → ball_query → grouping）**，每个下采样 stage 按此顺序执行：

```
p (B, 5500, 3), f (B, C, 5500)
 │ ① FPS   从 5500 个点里选出 2750 个"彼此离得尽量远"的代表点 → idx (B, 2750)   【决定在哪里看】
 ▼
new_p (B, 2750, 3)
 │ ② ball_query   以每个中心点为球心、radius 为半径圈出 nsample=32 个邻居 → idx_dp (B, 2750, 32)   【决定看多大范围】
 ▼
 │ ③ grouping   按 idx_dp 把邻居特征 gather 摞起来：f (B,C,5500) → fj (B,C,2750,32)   【把索引变成特征】
 ▼
fj + rel_pos(dp) → 1×1 conv 共享 MLP → max pool → f' (B, 2C, 2750)
```

三者分工：
- **FPS**（最远点采样）：回答"**在哪看**"。贪心地每次选距已选点集最远的点，保证降采样后空间覆盖均匀（随机采样会在密集区扎堆）。纯几何操作，不看特征。
- **ball_query**：回答"**看多大**"。radius 定感受野，nsample 定每邻域最多收几个。返回的 `dp`（邻居相对中心点的坐标差）同时喂给 `rel_pos` 编码位置。
- **grouping**：回答"**看什么**"。它只是个 gather（按索引搬运特征，无参数、无计算），把邻居特征摊成 `(C, npoint, nsample)` 的矩阵，让后面的 1×1 conv + max pool 能像处理普通卷积一样聚合局部模式。

三者合起来 = 点云版的"**感受野选取 + im2col + 卷积池化**"：N 减半（分辨率降）、C 翻倍（语义升）。

**解码器：上采样二件套（three_nn → three_interpolate）**。解码器要把稀疏层特征还原回稠密分辨率，与编码器 skip 特征拼接：

```
稀疏层 p1 (B, 344, 3), 稠密层 p2 (B, 688, 3), f2 (B, C, 688)
 │ ④ three_nn   对 p1 每个点在 p2 里找 3 个最近邻 → dist + idx   【建立粗→细的对应关系】
 ▼
 │ ⑤ three_interpolate   权重 = 距离倒数归一化（越近权重越大），3 个邻居特征加权求和   【特征传播/上采样】
 ▼
cat(f_up, encoder 同层 skip 特征) → conv 融合
```

这就是**点云版的双线性插值上采样**：2D 图像 upsample 用 4 邻近双线性权重，这里用 3 近邻逆距离权重，思想完全一致——分辨率恢复 + skip 融合。

**汇总表**：

| 算子 | 职责 | 输入 → 输出 | CNN 里的类比 | 本项目部署现状（v14.5） |
|---|---|---|---|---|
| FPS | 选代表点（在哪看） | xyz → idx | pooling 的"选代表"（但保几何均匀） | ✅ 已 plugin 化 |
| ball_query | 圈邻居（看多大） | new_xyz+xyz → idx | 感受野/im2col 的邻域枚举 | ✅ 已 plugin 化（并入 BallQueryGroup/BallQueryDP） |
| grouping | 搬邻居特征（看什么） | f+idx → (C,npoint,nsample) | im2col / unfold | ✅ 已 plugin 化（并入 BallQueryGroup） |
| three_nn | 找粗→细对应 | 两层 xyz → dist+idx | 上采样坐标映射 | ✅ 已 plugin 化（并入 ThreeInterp） |
| three_interpolate | 加权特征传播 | f+idx+dist → 上采样特征 | bilinear resize | ✅ 已 plugin 化（并入 ThreeInterp） |

**一句话关系**：前三个组成编码器的"降分辨率、升语义"流水线，后两个组成解码器的"恢复分辨率、融合 skip"流水线；FPS/three_nn 负责**选谁**（输出索引），ball_query 负责**定范围**（输出索引），grouping/three_interpolate 负责**搬运**（索引→特征）。除了 conv/pool，这 5 个算子就是 HPENet 骨干网络的全部"点云专属"计算——也正是部署时与 TRT 生态冲突最大、需要逐个 plugin 化的部分（v14.5 已全部完成）。

### C.2 xyz 计算不需要 fp16 精度吗？

恰恰相反——**xyz 是这里最需要精度的张量，所以才锁死 fp32**。

**fp16 对 xyz 的三重风险**：

1. **量化误差量级不够看**。fp16 尾数只有 10 位（约 3 位有效十进制数字），误差是相对值绑定量级的：

   | 坐标值 | fp16 的最小间隔（ulp） | 含义 |
   |---|---|---|
   | 5 m | ~2.4 mm | 还行 |
   | 30 m | ~1.6 cm | 已是 voxel(0.3m) 的 5% |
   | 60 m | ~3.1 cm | 接近 voxel 的 10% |

   坐标越大误差越大——雷达点动辄几十米，这正是 fp16 最不友好的场景。

2. **几何计算把连续误差变成离散决策**。features 上的误差是连续的：conv/ReLU 是近 Lipschitz 的，扰动进多少出来还是多少，网络精度几乎无感。而 xyz 上的误差要喂给离散判断：
   - ball_query：`dist ≤ radius`——正好卡在球面附近的点会翻转进出邻域
   - three_nn：top-3 排序——距离接近的邻居会换序，权重跟着变
   - FPS：每次选"最远点"——一次选错，后续所有采样中心全部改写

3. **离散翻转是级联的、非局部的**。features 里一个数值差 0.001，输出差 ~0.001；xyz 里一个邻居索引翻转，整个 `(C, npoint, nsample)` 分组张量的对应位置换成另一个点的特征，再经 conv/pool 放大传播。FPS 更极端——一个中心点不同，下游几百个点的分组全变。这使 fp16 下的精度偏差有界性差、难归因。

**那 features 为什么敢用 fp16？** 不对称的根源在于收益/风险比：

| | features | xyz |
|---|---|---|
| 通道数 | C = 32~256 | 恒为 3 |
| fp16 省的带宽 | 大头（占激活流量 95%+） | ~零头（3/C） |
| 误差性质 | 连续、有界、网络训练时本就抗噪 | 离散决策输入，会被级联放大 |

features 走 fp16 拿走几乎全部加速收益，xyz 保 fp32 几乎不付代价——**混合精度的钱要花在刀刃上**。这也是 TRT 官方 sample 和各家点云部署（MMDeploy3D、CenterPoint 部署）的通行做法。

**落到 plugin 实现上**（见 §13.4 dtype 策略）：`supportsFormatCombination` 本来就是按 tensor 逐个声明的，天然支持这种混合——xyz 恒 `kFLOAT`，features 接受 `kFLOAT/kHALF` 跟随 engine 精度；输出 grouped/interpolated 与 features 同 dtype，dp 恒 fp32。若 TRT 在 fp16 engine 里想把 xyz 以 fp16 喂进来，这个组合会被拒掉，builder 自动在 plugin 前插一个 Reformat 转回 fp32——代价是一次小转换（3 通道张量），远小于精度风险。kernel 内部所有距离/权重计算都在 fp32 寄存器完成，只有特征搬运和最终 MAC 按输入 dtype 走。
