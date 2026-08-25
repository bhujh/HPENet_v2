# Learnings — gridballquery-trt-plugin

Conventions, patterns, and successful approaches discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## 2026-08-22 Task3: GridBallQuery 导出侧接线完成
- 新 op 照抄 ballquerygroup/dp_op 结构，仅 op_type + voxel_size_f(-1.0=插件默认 radius) 不同；grid kernel 无 python 绑定 → forward 全占位（CPU/CUDA 同），数值验证全在 TRT 层。
- bq_algo 走"闭包内选工厂"路线：_make_sa_group_forward / _make_invresmlp_group_forward 加 bq_algo 参数，默认分支逐字节等价原逻辑；注意闭包捕获变量（grouper 等）须在分支前赋值。
- 回归方法：改前/改后各导出一次默认档，逐节点比较 (op_type, domain, inputs, outputs, attributes) → 完全一致；grid 档 628 节点与默认档相差 0，仅 8 个 op_type 改名（BQG 4 + BQD 4）。
- 产物：deploy/hpenet_v2_gridbq.onnx（fps_algo 默认 fps_cache_prune + bq_algo gridballquery）。证据：deploy/evidence/gridballquery-task3-export.md。
## 2026-08-22 task1: gridballquery_kernel.cu
- nvcc: -Xcompiler 后的 -Wall -Wextra 必须整体引号传入（`-Xcompiler "-Wall -Wextra"`），否则 nvcc fatal: Unknown option '-Wextra'。
- bq_dp_kernel/bq_gather_kernel 在 ballquerygroup_kernel.cu 匿名命名空间内，跨 TU 不可链 → 复制进新 .cu（文件头注明来源行号 13-66）。
- 单 block 前缀扫描：T 个元素严禁进 smem（32768×4B=128KB 超限）；做法=每线程连续 chunk 两遍读 counts（寄存器累加 + 写回时重读），warp shfl 扫描 + 32 float smem warp 部分和。sme 总 128B。
- 稳定放置：rank = O(N²) 前驱同桶计数（N≤10000 可接受），禁 atomicAdd 顺序依赖；counts 的 atomicAdd 是纯计数（序无关）允许。
- grid 语义对齐 ballquery：d2/radius2 表达式逐字复制（普通运算符），严格 <，半径内取索引最小 nsample 个，升序写入，空槽填最小索引候选，空球 memset 0。
- 哈希桶无探测 → 查询时每候选必须重算其 cell 验证 == 目标格，否则哈希碰撞跨格混入。
- per-thread 局部最大堆容量编译期上限 GRIDQUERY_MAX_NSAMPLE=256（nsample 超限 launcher 直接 no-op，任务2插件侧应 static_assert）。
- workspace 切片 256B 对齐：counts[T]+prefix[T]+start[T+1]+sorted[N]+idx_ws[M*S]；不足时 launcher 静默返回（TRT getWorkspaceSize 由任务2保证尺寸）。
- 编译：`nvcc -c deploy/trt_plugins/src/gridballquery_kernel.cu -o ... -Ideploy/trt_plugins/include -Xcompiler "-Wall -Wextra" -std=c++17` → 0 error 0 warning。
## 2026-08-22 Task2: GridBallQuery 插件类
- `static_cast<unsigned char* → int*`（跨类型的指针 adjust）非法，workspace 段偏移取指针必须 `reinterpret_cast<int*>(base + off)`。
- TRT 8.6 python 探针：`trt.get_plugin_registry().get_plugin_creator(name, "1", "")`（tests_fps_algos.py:109 模式）；`trt.Registry` 不存在，`reg[i]` 索引也取不到东西。import tensorrt 需 LD_LIBRARY_PATH 带 nvidia/cudnn/lib + TRT lib（libcudnn.so.8 找不到时）。
- 同进程 ctypes.CDLL 两次 dlopen 幂等（same handle=True），REGISTER_TENSORRT_PLUGIN 静态注册器不会重复注册报错。
- workspace 镜像公式：插件侧全用 configurePlugin 缓存的 profile 极值（maxN/maxM），launcher 内用运行时 n/m——maxN 基偏移 ≥ n 基，idx_ws 指到 maxN 基尾段是安全方向冗余。
- 运行时属性（nsample≤256）不能 static_assert → createPlugin/deserializePlugin 读出后检查、stderr 打日志 + 返回 nullptr 拒绝（比 launcher 静默 no-op 早失败）。
- 证据：deploy/evidence/gridballquery-task2-plugin.md（11/11 registry 探针 OK，构建 0 error 0 warning）。
## 2026-08-22 Task4: GridBallQuery 单元对拍（首次数值验证）全绿
- 31/31 PASS；判据③ 实测最大 diff dp=0 / grouped=0（fp32 与 fp16 完全一致）；bit 一致率 cnt≤S 218/218、cnt>S 21332/21332 均 100%；空球 4 行 grid 全 0。
- 现役 ball_query_kernel_fast 同样是 k 升序扫 + 首邻居 padding + cnt≥nsample break → cnt>nsample 行现役也取"索引最小 nsample 个"，与 grid 堆选语义天然逐 bit 一致（诊断两分支全一致的理论依据）。
- 现役空球行 idx 不写（未初始化内存既有 bug）：测试侧给输出 buffer 预清零（torch.zeros）即可规避 dp/gather 越界读，且该行跳过对比只验 grid 全 0。
- fp16 Group engine 路线：ONNX 里 features 输入声明 FLOAT16 + FP16/OBEY flag；grouped 输出仍声明 FLOAT（getOutputDataType 恒 kFLOAT），半精度 gather 在插件内走（supportsFormatCombination grouped==features），TRT 出口 reformat 回 fp32——build 快且稳，无 binding 类型冲突。
- 测试脚本坑：f-string 内单引号 r.get('pass') 会躲过 sed 的双引号替换模式——重命名 dict key 时务必 grep 原词；`max(...) or None` 会把合法的 0.0 吞成 None。
- 矩阵耗时 397s（python 参考逐行 flatnonzero 循环为主），略超 5 分钟目标但可接受；engine 缓存后每组合实际 4 个静态 engine。
## 2026-08-22 Task5: GridBallQuery 整网 E2E 验证
- fp32 整网 grid vs 默认档：58 真实帧 + 合成 N{2024,4096,10000} 全部 **bit 级一致**（maxdiff=0, agree=100%）——插件语义经全网络逐 bit 等价的最强证据。
- fp16 整网 diff ~1e0 量级超标但非插件问题：同 ONNX 双 build bit 一致（builder 确定）+ fp32 全网 bit 一致 + 插件级 fp16 bit 一致 → 三重证据指向 TRT FP16 tactic/Myelin 融合对图结构敏感（节点类型不同→融合边界不同→累加序不同）。mIoU 仲裁 0.04pp 通过。
- trt_inference.py 的 0.2 split 与 RadarClassi 的 83/17 seed=100 不一致；eval 脚本用后者口径（sorted listdir→seed100 shuffle→[int(0.83n):]，339 文件→58 测试帧）。
- engine mtime 顺序可作 provenance 初筛：现役 fp16(10:27) 早于 plugin.onnx(10:31) → 重建对照，勿信 mtime 相近即同源。
- preprocess_test 的 voxelize shift i=0 单 sub-cloud 即可跑 mIoU（label 取 idx_part 对应子集），不需要 scatter merge；pad 复制末点后 logits 裁回 N_true。
## 2026-08-22 Task6: nsys 性能对比（负结论）
- **GridBallQuery 全面更慢**：搜索段慢 8.9×(N=2024)/9.4×(4096)/13.5×(10000)，整网 gpu_only 慢 2.4~3.5×，差距随 N 扩大。不建议替换。
- 主导瓶颈是 grid_query（占搜索段 82~87%，随 N 超线性）而非预判的 build_place（O(N²) place 占 12~17%，但单项已≈现役搜索全程）。
- 现役 ball_query_kernel_fast 近线性（k 升序早 break 稀疏剪枝极有效）；grid engine fp16 整网反而略慢于 fp32（瓶颈在 fp32 插件 kernel + reformat 开销）。
- nsys 2022.4 无 cuda_gpu_kern_sum，报告名是 gpukernsum/nvtxkernsum（--help-reports 列表）；TRT 层自动 NVTX（nvtxkernsum 按 layer 出 kernel 表）可用于定位调用点。
- per-inference kernel 耗时 = gpukernsum Total / 序列数（warmup+gpu_only+e2e=220），contamination 同比可忽略。
- 基准脚本 deploy/bench_gridballquery.py：cuda.Event 双口径（gpu_only=仅 executeV3；e2e=pinned H2D+execute+D2H），NVTX warmup/inference 两 range；输入真实帧 pad/裁剪+合成轮换。证据 deploy/evidence/gridballquery-nsys.md（12 份 .nsys-rep）。
- [T7 文档] plugin.md 追加 §15「GridBallQuery 档」（v16，编号接续 §14，纯 append 未动既有内容含风险表）；AGENTS.md DEPLOY 节两处增补（插件 9→11 + --bq_algo 行）。grep 'GridBallQuery' plugin.md = 7。
- 文档约束：plugin.md 有大量未提交既有改动（v15.x），只能文件末尾追加；新风险行放新章节内表格（§15.7），不改 §10 旧表；AGENTS.md 只改 DEPLOY 节那一行+加一行。
- 引用口径：全部数字逐字取自 deploy/evidence/gridballquery-{unit,e2e,nsys}.md 并标注文件名，不虚构。
