# CPP_trt3 部署代码整改计划（代码审查修复）

> 创建：2026-08-26 | 状态：**执行完成（除 R1 按用户指示跳过）**
> 前置：Oracle 两轮审查 —— ① `deploy/CPP_trt3` 代码审查发现 **2 处 BUG + 8 处 ERROR + 8 处 REDUNDANT**；② 修复方案二次审查，判定 **5 项可直接执行、4 项需修正、4 项遗漏**。本计划为二次审查修订后的最终可执行版本。**③ Momus 计划评审：APPROVE-WITH-FIXES（4 处小偏差已修正）。④ Oracle 独立复核（第 3 次）：APPROVE-WITH-FIXES——发现审查期间 3 个源文件被并行修改（logger.h 18:12 / main.cpp 18:13 / wrapper.h 18:21），引入 [P0] broken build + [P2] help 六处不一致 + [P1] wrapper voxel_size 0.3f 漂移，已全部纳入本计划。**
> ✅ **broken build 已修复**：`logger.cpp` TRTLogger 定义已 `#if 0` 禁用、`TrLogger` 保留、编译通过（P0-0 完成，TODO-0 已 [x]）。
> ✅ **⑤ 计划 vs 代码一致性审查（Oracle，最终）：CONSISTENT** —— 全部 18 个已勾选项与代码逐项吻合、无新 bug；发现 3 处 P3 级残留已补修：①wrapper.h 三个函数的假 `@param handle`（process_file/process_inmemory/infer_all_radars）已删；②wrapper.cpp L199 `ONNX initialization failed` 文案改 `TensorRT`；③trim_transpose.cu 补 B4 排除理由注释。
> 约束：**零 git 操作**（用户明确）；改推理逻辑项（R1/E6）落地后必须跑 ti10 对拍验证 pred 逐点一致 + acc 0.9578。
> 风险分级：P0 生产阻塞 / P1 高价值重构 / P2 正确性加固 / P3 死代码+文档清扫。

---

## 一、背景

`deploy/CPP_trt3`（当前部署工作流，TensorRT C++ 推理管线）经 Oracle 审查：**核心推理路径正确性扎实**（offset/chunk 算术、channel-major 布局、fp16 转换、stream-FIFO 排序均核对无误，无影响推理结果的缺陷），问题集中在**外围**——CLI 越界、wrapper 调试残留、死代码/重复代码、kernel 错误检查缺失。

原始审查结论（供追溯）：
- **BUGS**：B1 argv 越界读；B2 wrapper 生产接口硬编码调试写入；B3 全局句柄锁不一致；B4 kernel launch 无错误检查；B5 float→int32 越界 UB；B6 loadv2 不校验 PLY format。
- **ERRORS**：E1 min_n 默认 2024 vs help 1024；E2 num_files 默认 100 vs help -1；E3 nvtx VOXELIZE 归属；E4 注释 17% vs 80%；E5 has_label 未置位（有意）；E6 total_src int 溢出；E7 CHECK_CUDA abort vs 库形态异常策略矛盾。
- **REDUNDANT**：R1 双入口 ~190 行重复；R2 d_logits_ 未使用；R3 死代码；R4 trim_padding 死代码；R5 TRTLogger 未使用；R6 main.cpp 重复文件切分；R7 preprocessor 先建后拷；R8 wrapper 头文件注释误导。

---

## 二、修复方案

<!--
### P0-0 — 修复构建（前置，最高优先级）【已取消：用户自行处理 logger.h/logger.cpp】

#### 新增：删除 logger.cpp 的 TRTLogger 定义（修 broken build）
**位置**：`src/logger.cpp:13-120`

审查期间 `include/logger.h`（mtime 18:12）的 TRTLogger 类定义被注释（L17-47），但 `src/logger.cpp` 仍活跃定义 TRTLogger 成员（L13-120）。现有二进制 14:50 早于头文件修改 → **下一次 `make` 重编 logger.cpp 即报 "TRTLogger not declared"**。全工程无其他引用（main/pipeline 均用 `TrLogger`）。

**改法**：删 `logger.cpp:13-120` 的 TRTLogger 全部定义（severityToString/logWorker/ctor/dtor/log），保留 `TrLogger`（L126-147）。这是所有批次的前置条件——批次①之前必须先做，否则任何重编验收都失败。

---
-->

### P0 — 生产阻塞级（可直接执行）

#### B2：用 `#if 0` 包裹 wrapper 生产接口的调试残留
**位置**：`src/trt_inference_wrapper.cpp:330-354`（`trt_ai_infer_and_update`）

调试残留**散布在真实逻辑之间**，须**分段** `#if 0 ... #endif` 包裹（**不可整段包裹**——L342 推理、L353 写回是真实逻辑必须保留）：
- `#if 0` 块①：L330-335（写初始点云 `write_annotated_ply<float>` + cerr "first write_annotated_ply"）
- `#if 0` 块②：L338-341（冗余 handle 判空 + cerr "handle is null"；L313 已查，冗余）
- `#if 0` 块③：L343（单行 cerr "inference"）
- `#if 0` 块④：L345-350（写预测 `write_annotated_ply<int>` + cerr "second write_annotated_ply"）
- `#if 0` 块⑤：L354（单行 cerr "update_predictions_to_cdi"）

**保留（真实逻辑）**：L342（`process_pointcloud` 推理）、L353（`update_predictions_to_cdi` 写回）。

如需离线调试，改对应 `#if 0` 为 `#if 1` 或 `#ifdef DEPLOYAI_DEBUG_DUMP`。`write_annotated_ply` 模板本身仍被 `trt_pipeline_process_file`(L237) 使用，头文件声明保留。

#### B1：CLI `--key value` 形式越界读 argv
**位置**：`src/main.cpp:117-127`

`parse_args` 开头加取值 lambda，替换 11 处 `argv[++i]`：
```cpp
auto next_value = [&]() -> const char* {
    if (i + 1 >= argc) {
        std::cerr << "Error: missing value for argument '" << arg << "'\n";
        std::exit(1);
    }
    return argv[++i];
};
// else if (arg == "--engine") cfg.engine_path = next_value();
// else if (arg == "--num_files") cfg.num_files = std::stoi(next_value());
// ... 其余 9 处同理；--benchmark（布尔）不改
```

---

### P1 — 高价值重构

#### R1：合并 process_pointcloud / process_file 双入口
**位置**：`src/pipeline.cpp:83-272`（process_pointcloud）与 `:278-~493`（process_file）

抽私有核心 `process_pointcloud_impl(const PointCloud& pc)`（现 process_pointcloud 主体，已是 const 语义、内部用局部 coord 副本），两入口薄封装。**三条细则（Oracle 修订，必守）**：
1. **Step 9 dump 必须留在 `process_file` 包装层**——`fs::path(ply_path).stem()` 依赖文件路径，impl 只有 PointCloud 拿不到 stem，误搬进 impl 会编译失败或 dump 到错误文件。
2. **nvtx 归属**：推荐 impl 统一带 nvtx range（CDI 路径零成本获得剖析能力）；若 impl 不带则 process_file 失去分段粒度。
3. **coord-shift 统一为局部拷贝**（现 process_pointcloud 已是，process_file 改对齐）：行为保持（process_file 就地修改的 pc 是函数内部加载副本，调用方不受影响），代价仅 +1 次 vector 拷贝，可忽略。注意 `total_src` 统计在 process_file 里位于 VOXELIZE range 内（L314-318），拆分后 range 边界需重新对齐。

#### R2：删除未使用的 `d_logits_`
**位置**：`src/pipeline.cpp:74-75` + `include/pipeline.h:95`

删除构造分配 + 成员声明。全工程无引用（GPU logits 走 `TrInference::d_output_`），删除安全。

#### 遗漏项 #1（升 P1）：trt_pipeline_create 全局句柄竞态
**位置**：`src/trt_inference_wrapper.cpp:180-192`

`trt_pipeline_create` 在无锁状态下读 `g_trtinfer_handle`（L180）并调 `trt_pipeline_destroy()`（destroy 内部才拿锁）→ TOCTOU 竞态。

**具体改法（用户确认，含 E1 顺带修）**：
```cpp
DEPLOYAI_LIB_API TensorrtInferencePipeline_C* trt_pipeline_create(
    const char* onnx_path, const char* stats_json_path) {
    if (!onnx_path || !stats_json_path) return nullptr;
    try {
        std::lock_guard<std::mutex> lock(g_mutex);   // 全程持锁，消 TOCTOU
        // 重新初始化：内联销毁旧句柄（不调 trt_pipeline_destroy，避免死锁）
        if (g_trtinfer_handle) {
            delete g_trtinfer_handle;
            g_trtinfer_handle = nullptr;
        }
        auto handle = std::make_unique<TensorrtInferencePipeline_C>();
        handle->cpp_pipeline = std::make_unique<InferencePipeline>(
            std::string(onnx_path), std::string(stats_json_path), glogger,
            2024, 10000, 0.02f);   // voxel_size 0.3f→0.02f（匹配训练口径，E1）
        g_trtinfer_handle = handle.release();
        return g_trtinfer_handle;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: ONNX initialization failed: " << e.what();
        return nullptr;
    }
}
```

**要点**：①`lock_guard` 移到 try 内逻辑最前、覆盖读检查 + 销毁 + 赋值全程；②销毁用**内联** `delete` + 置空（`std::mutex` 非重入，持锁调 `trt_pipeline_destroy()` 会死锁）；③顺带修 voxel_size 0.3f→0.02f（E1）；④全局单例句柄语义（多使用者场景需另行设计，本次仅修竞态）。

---

### P2 — 正确性加固

#### B4：kernel 错误检查（**含 sync 隔离修正，最高风险项**）
**位置**：`src/scatter_mean.cu:73-92`、`src/fnv_hash.cu:60-70`、`src/voxelizer.cu`（thrust 调用）

**修正要点（Oracle 修订，必守）**：
1. **同步版 `CHECK_LAST_CUDA()`（内含 `cudaDeviceSynchronize`）只能进 debug 构建**（`#ifdef` / 运行时 flag 隔离）。`launch_scatter_mean_kernel` 位于 pipeline 软件流水尾部、Step 6 D2H 之前，注释明确"文件内唯一 sync"——生产路径插入同步会破坏零 sync 流水、每文件延迟显著上升。
2. **NO_SYNC 版（仅 `cudaGetLastError`）命中必须 abort/throw**——`cudaGetLastError` 会清除 sticky error，若只打日志会吞掉后续本应发现的错误。
3. **补 `scatter_mean.cu:73-74` 两处 `cudaMemsetAsync` 返回值检查**。
4. **显式排除 `launch_trim_transpose_kernel`**——B4 给 kernel 加错误检查时**不要**给这个 kernel 加（它每个子云推理后调用一次，加 `CHECK_LAST_CUDA` 内含的 `cudaDeviceSynchronize` 会让每 chunk 都同步，破坏 pipeline「文件内唯一 sync」的零同步流水、每文件延迟显著上升）；在代码注释写明排除理由，防止执行者"顺手补全"。
5. Voxelizer 路径（默认流 0 + 后续同步 `cudaMemcpy` 已隐式同步）加检查近乎免费，可行。

#### B5：float→int32 越界 UB
**位置**：`src/fnv_hash.cu:53-55`

```cpp
long long vx = static_cast<long long>(floorf(pt[0] / voxel_size));
int32_t ix = static_cast<int32_t>(vx);  // ll→int32 为实现定义，不再是 UB
```
**注释必写**：clamp 是"缓解非对齐"——Python 侧 `fnv_hash_vec` 用 int64 坐标，极端坐标（|coord/voxel|>2^31）hash 与 Python 不一致；实际雷达数据不触发。可顺带对 `!isfinite` 输入早退。

#### B6：loadv2 校验 PLY format
**位置**：`src/ply_reader.cpp:137-221`（`loadv2`）

`loadv2` 手工 `getline` 逐行读数据，仅支持 ASCII；binary PLY 会通过 parse_header 后在 L193 报误导性 "PLY truncated"。
**实现机制（Oracle 修订）**：`tinyply::PlyFile` 常见版本无公开 format 访问器 → 在 parse_header 前自行 sniff 头部（重开文件 / 先读原始行检查 `format ascii 1.0`），非 ascii 即 `throw std::runtime_error("only ascii PLY supported")`。**仅限 `loadv2`**（`load()` 走 tinyply `read()` 本身支持 binary，勿加限制）。

#### E6：total_src int 溢出（**改穿调用链，勿半截修改**）
**位置**：`src/pipeline.cpp:119` 与 `:314`；关联 `include/scatter_mean.h:28`、`src/scatter_mean.cu:23,79`、`include/trim_transpose.h:19`

只改 `total_src` 变量不够，真正溢出点在 kernel 内 `num_src * C` 的 int 乘法。**一次改穿**：
1. `pipeline.cpp` 两处 `int total_src` → `size_t`，`offset` → `size_t/int64_t`。
2. `launch_scatter_mean_kernel` 签名 `int num_src`（`scatter_mean.h:28`）→ `int64_t/size_t`；kernel 内 `num_src * C`（`scatter_mean.cu:23`）、`total_threads`（`:79`）与 **`int idx = blockIdx.x*blockDim.x+threadIdx.x`（`:22`）及其 `idx >= num_src*C` 比较**同步升 64 位（否则 idx 先在 int 域溢出）；**launch 侧 `grid_size = (num_src*C + 255)/256` 的除法表达式也须在 64 位域计算**（否则 launch 前就截断）；`trim_transpose` kernel 内部 idx 行同查。
3. `launch_trim_transpose_kernel` 的 `int offset`（`trim_transpose.h:19`）→ `int64_t/size_t`，同步改 `.cu` 与调用点。
4. `idx_staging_.assign(total_src, 0)` 与 `d_idx.upload` 的 size_t 转换已兼容。
若不愿改穿调用链，则**本项标注"理论性、暂不做"**（实际 max_n=10000 下无溢出），避免半截 signedness 混用引入隐式收窄告警。

#### E7：CHECK_CUDA 拆 THROW / ABORT
**位置**：`include/logger.h:60-68`

拆两宏：
- `CUDA_CHECK_THROW`（throw `std::runtime_error`）——业务路径（`upload/download/memset` 等，**这些方法就在 cuda_utils.cu 里**）用，异常穿过 `extern "C"` 边界被 wrapper 的 try/catch 兜底。
- `CUDA_CHECK_ABORT`（`std::abort`）——仅 `CudaBuffer::~CudaBuffer`、`CudaStream` dtor 等 noexcept 上下文用（dtor 默认 noexcept，throw 会 terminate）。

⚠️ **宏分配（Metis 修订，必守）**：cuda_utils.cu 内 **dtor/cudaFree 用 ABORT、其余业务方法（ctor 的 cudaMalloc、upload/download/memset）用 THROW**——不可写「本文件禁用 THROW」（业务方法就在本文件，禁用会与 THROW 策略自相矛盾）。执行时对 14 处 CHECK_CUDA 逐调用点分配，列出映射表。

---

### P3 — 死代码清理 + 文档修正（可选批次，纯删除/文档，低风险）

| 项 | 位置 | 改法 |
|---|---|---|
| R3 | `trt_inference_wrapper.cpp:35-65` `convert_cdi_to_pointcloud`(v1) | 用 `#if 0 ... #endif` 包裹（已被 `_v2` 取代，无调用），不物理删除 |
| R3 | `trt_inference_wrapper.cpp:133-171` 注释掉的旧 create/destroy | 已注释，保留不动 |
| R3 | `ply_reader.cpp:224-305` 注释掉的 `write_annotated_ply<int>` + `write_annotated_ply<float>` 两个特化 | 已注释，保留不动 |
| R4 | `subcloud_utils.cpp:148-173` `trim_padding` + `subcloud_utils.h` 声明 | 用 `#if 0 ... #endif` 包裹（GPU `trim_transpose_kernel` 已取代），不物理删除 |
| R5 | `logger.cpp:13-120` 的 `TRTLogger` 成员定义 | **已取消（用户自行处理）**：用户正在改 logger.h/logger.cpp，此清理由用户完成，不在本计划执行范围 |
| R6 | `main.cpp:204-220` 重复文件切分 | 给 `InferencePipeline` 加 `static list_test_files(dir, num_files)`，`process_directory` 与 main 共用 |
| R7 | `preprocessor.cpp:67-117` pos 中间向量 + memcpy | 直接在 `result.pos` 上计算，去掉中间 `pos` 向量 + L117 memcpy。**注意 pos 的全部引用点（用户已确认）**：L67 创建、L68-74 填充（coord_part−mean_pos）、L78-85 gravity-zero（读/写 pos 的 z 列）、L111 height_norm（读 pos 的 z）、L117 memcpy——须全部改为直接读写 `result.pos`（L67 处改 `result.pos.resize(N*3)`），**不能只改 L67-74 而漏掉 L78-85/L111**（否则 `pos` 未定义导致编译失败） |
| R8 | `trt_inference_wrapper.h` 通篇"ONNX"误写（L3 `@file onnx_c_api.h`、L4/L27/L33/L41/L45/L75/L168）+ L78 `@param handle` 无参文档 + L35-43 `trt_pipeline_create` 假 @param | 注释改"TensorRT"；删 L78 无参 `@param handle`；删 create 文档的 5 个假 @param（实际签名 2 参） |
| E1 | `trt_inference_wrapper.cpp:189`（`trt_pipeline_create` 内）硬编码 `(2024, 10000, 0.3f)` 三元组 | **方向已定（用户确认）**：voxel_size `0.3f` → `0.02f`（匹配训练口径 20260825-161134 cfg.yaml:21；`trt_ai_infer_and_update` 本身无 voxel_size，经全局句柄间接使用此处值）→ 改命名常量统一为 0.02f；②`trt_pipeline_create` 头文件声明仅 2 参但文档列了 4 个假 `@param`（min_n/max_n/voxel_size/seed）→ 修文档 |
| E2 | `main.cpp:31` `num_files=100` vs help L56 写 -1 | **方向已定（用户确认）**：改 help 文本为 100（默认值保持 100 不变） |
| E3 | `pipeline.cpp:314-318` nvtx VOXELIZE pop 位置 | `total_src` 统计移到 `nvtxRangePop() // VOXELIZE` 之后（若 R1 已重排则随 R1 一并处理） |
| E4 | `pipeline.h:72` + `pipeline.cpp:509` 注释"后 17%" | 两处都改"后 80%（`test_start = n_total*0.2`）" |
| E5 | `trt_inference_wrapper.cpp:93` 写 pc.label 未设 has_label | **有意为之**（label 存原始 valid 字段，非 ground truth，设 has_label 会让 accuracy 用错标签）→ 仅加注释说明，不改逻辑 |
| E8 | `main.cpp:46-74` help 文本 vs `CLIConfig`(L24-41) 默认值六处不一致 | 一次性对齐：`--max_n` 6000→10000、`--voxel_size` 0.1→0.02、`--engine`/`--stats`/`--data_dir`(radarfull→radarfullwl)/`--output` 帮助路径改为实际默认（连同 E2 的 num_files 共七处） |

---

## 三、TODOs

- [x] 0. **P0-0（已取消，用户自行处理）**：删 `logger.cpp:13-120` TRTLogger 定义修 broken build——用户正在改 logger.h/logger.cpp，此任务由用户完成
- [x] 1. **P0-B2**：`#if 0` 分段包裹 wrapper `trt_ai_infer_and_update` 5 段调试残留（L330-335/338-341/343/345-350/354，**保留 L342/L353 真实逻辑**）——验收：CDI 路径不再写 `/CPP_trt1/output/`，重编通过
- [x] 2. **P0-B1**：`parse_args` 加 `next_value` lambda，替换 11 处 `argv[++i]`——验收：`--engine`（无值）报错退出而非越界
- [ ] 3. **P1-R1**：抽 `process_pointcloud_impl`，两入口委托，dump 留包装层、coord-shift 统一局部拷贝——验收：ti10 pred 逐点 bit 级一致 + acc 0.9578
- [x] 4. **P1-R2**：删 `d_logits_`（pipeline.cpp:74-75 + pipeline.h:95）——验收：重编通过，无引用残留
- [x] 5. **P1-遗漏#1**：`trt_pipeline_create` 全局句柄持锁 + 内联销毁 + voxel_size 0.3→0.02——验收：create 全程持 g_mutex、内联 delete 不调 destroy、voxel 0.02f
- [x] 6. **P2-B4**：kernel 错误检查——新增 `CUDA_CHECK_LAST`（仅 cudaGetLastError 不同步，命中即 throw）+ scatter_mean memsetAsync 用 `CUDA_CHECK_THROW` + scatter/fnv_hash/voxelizer launch 后 `CUDA_CHECK_LAST`；**排除 trim_transpose**（每 chunk 调用，未加）——验收：重编通过、acc 0.9595 无变化、零 sync 流水未破坏
- [x] 7. **P2-B5**：fnv_hash float→int32 改 long long 中转 + isfinite 早退——验收：重编通过，常规坐标 hash 不变
- [x] 8. **P2-B6**：loadv2 加 format sniff（仅 loadv2）——验收：binary PLY 报清晰错误
- [x] 9. **P2-E6**：走「暂不做」分支——`int total_src` 加注释标注「int 足够（max_n=10000 下无溢出）」，不改穿调用链（避免 int64 CUDA kernel 性能下降）
- [x] 10. **P2-E7**：CHECK_CUDA 拆 `CUDA_CHECK_THROW`（12 处业务路径）+ 保留 `CHECK_CUDA`（4 处 noexcept dtor/move）——验收：重编通过，业务路径异常可被 wrapper 捕获
- [x] 11. **P3**：R3/R4（`#if 0` 包裹死代码）/R6（list_test_files 消除重复切分）/R7（pos 直接写 result.pos）/R8（ONNX→TensorRT 文档）+ E1②（删假 @param）/E2（help num_files=100）/E3（nvtx pop 提前）/E4（注释 17%→80%）/E5（has_label 注释）/E8（help 六处对齐）——验收：重编通过、acc 0.9595 无变化、无 '17%' 残留

---

## 四、验收口径

1. **推理正确性（R1/E6 等任何动推理逻辑的项）**：**改动前先跑一次生成 ti10 基线 dump（并重跑一次确认基线自身 bit 确定，排除 scatter_mean 原子加本征非确定的误判）**，改动后 `./hpenet_trt_infer --num_files 10 --voxel_size=0.02 --dump-prefix /tmp/xxx` 对照改前 dump，**pred 逐点 bit 级 0 mismatch + acc 逐文件 4 位小数逐字符一致（均值 0.9578）**；logits 不做 bit 级判据（scatter_mean 原子加本征非确定）。
2. **B2**：`trt_ai_infer_and_update` 路径不再产生 `/CPP_trt1/output/` 文件、无 cerr 调试输出。
3. **B1**：末位参数缺失时打印清晰错误并退出（exit code 1），无越界/段错误。
4. **B4**：release 构建下 `launch_scatter_mean_kernel` 不引入任何 `cudaDeviceSynchronize`（nsys 复测 sync 次数不增）。
5. **E7（故障注入）**：临时在某业务路径 `cudaMemcpyAsync` 传非法指针，确认异常被 wrapper 的 try/catch 捕获并返回 -3（而非 terminate）。
6. **R1（nvtx 基线）**：合并后 `process_pointcloud`（CDI 路径）与 `process_file` 的 nvtx range 集合可能变化，nsys 前后对比时注明基线口径差异，不以 nvtx 分段作为对拍判据（对拍只用 pred/acc）。
7. **零 git 操作**：全程不 commit/push/reset。

---

## 五、止损条款

- **R1（合并双入口）**：合并后 ti10 pred 不一致 → 最多 1 轮排查，仍不一致则**回退保留双入口**（本项降级为"仅删 R2 d_logits_"），失败证据如实记录。
- **E6（改穿调用链）**：若改穿引入编译告警或对拍回归 → 回退并**标注"理论性，暂不做"**（实际数据量下无溢出）。
- **B4（sync 隔离）**：若 release 构建误引入 sync → 立即回退该宏改动，重新隔离到 debug 构建。
- 任一批次完成即单独重编 + 冒烟（`--num_files 2`），不攒到末尾一次性验证。

---

## 六、执行顺序与工作量

| 批次 | 内容 | 工作量 | 风险 |
|---|---|---|---|
| ~~⓪~~ | ~~P0-0（删 logger.cpp TRTLogger 修构建）~~ 已取消（用户自行处理） | — | — |
| ① | P0（B2+B1）+ P1-R2 | Quick（<30min） | 无（纯删除+越界检查） |
| ② | P1-R1 + P1-遗漏#1 | Short（1-2h，需 ti10 对拍） | 中（动推理逻辑） |
| ③ | P2 全部（B4/B5/B6/E6/E7） | Short（1-2h，需回归） | 中（B4 sync/E6 改穿） |
| ④ | P3 全部 | Quick（<1h） | 无（纯删除+文档） |

**建议**：⓪ 已由用户自行处理；①④ 可先行；②③ 严格按止损条款执行、逐项对拍。
