# cuDNN 段错误排查与修复记录

**日期**: 2026-06-03 | **环境**: HPENet V2, NVIDIA L20 × 8, Driver 590.48.01

---

## 1. 现象

执行 `script_me/main_segmentation_train.sh` 后，训练在第一个 batch 的模型 forward 阶段崩溃：

```
Fatal Python error: Segmentation fault
  File "torch/nn/modules/conv.py", line 306 in _conv_forward
  → hpenetv2.py:157  (SetAbstraction.forward: Conv1d(4,32))
```

---

## 2. 排查方法

### 2.1 快速定位崩溃点

```bash
# 环境变量捕获精确崩溃位置
CUDA_LAUNCH_BLOCKING=1 python -X faulthandler examples/segmentation/main.py ...
```
- `CUDA_LAUNCH_BLOCKING=1` → CUDA 操作同步执行，避免异步崩溃无法定位
- `-X faulthandler` → Python 崩溃时打印 C 层调用栈

### 2.2 最小复现 — 隔离变量

训练脚本耦合度高，逐层拆解为最小复现：

| 步骤 | 测试内容 | 结果 |
|------|---------|------|
| ① | `torch.randn(8,4,3000).cuda()` + `Conv1d(4,32)` | 💥 |
| ② | `torch.randn(8,4,3000).cuda()` + 基础运算 (add/matmul/relu) | ✅ |
| ③ | 同①，`torch.backends.cudnn.enabled = False` | ✅ |
| ④ | `Conv2d(64,128)` 默认 cuDNN | 💥 |

**结论**: cuDNN 后端才是真凶，不是 CUDA 驱动/内存/数据。

### 2.3 环境差异化排查 — "之前能跑现在不行"

```
成功: May 9-14 (kernel 6.17.0-23)
失败: Jun 3     (kernel 6.17.0-29, 5月20日左右重启切换)
```

检查系统变更：
```bash
# 内核安装记录
ls -la /boot/vmlinuz-*
# apt 历史
cat /var/log/apt/history.log
# NVIDIA 驱动版本
nvidia-smi --query-gpu=driver_version --format=csv
# 系统 cuDNN 安装时间
ls -la /usr/local/cuda-11.8/lib64/libcudnn*8.9.7*
```

发现 5月18日安装了 cuDNN 8.9.7。

### 2.4 库加载路径追踪 — "到底用了哪个 cuDNN"

```bash
# 方法1：看 ldconfig 缓存
ldconfig -p | grep libcudnn

# 方法2：看 LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# 方法3：PyTorch 看加载版本
python -c "import torch; print(torch.backends.cudnn.version())"
# 8907 → 8.9.7

# 方法4：RPATH 追踪（LD_DEBUG 看搜索路径）
LD_DEBUG=libs python -c "import torch" 2>&1 | grep cudnn

# 方法5：ldd 看链接
ldd ...torch/lib/libtorch_cuda.so | grep cudnn
```

### 2.5 文件完整性校验

```bash
# 对比 conda 和系统 cuDNN 的 hash
md5sum conda_env/.../nvidia/cudnn/lib/libcudnn.so.8
md5sum /usr/local/cuda-11.8/.../libcudnn.so.8.7.0

# 结果：全部 DIFFER——conda 里的"8.7.0"实际是 8.9.7
```

---

## 3. 根因链

```
系统 cuDNN 8.9.7 于 2026-05-18 安装
    ↓
pip 缓存受污染（nvidia-cudnn-cu11 包的 .so 被 8.9.7 替换，包名仍标 8.7.0.84）
    ↓
5月20日系统重启 + 内核切换 (6.17.0-23 → 6.17.0-29)
    ↓
~/.bashrc 的 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 优先加载系统 cuDNN
    ↓
同时 conda 环境内的 cuDNN 也是 8.9.7（坏文件）
    ↓
cuDNN 8.9.7 的 Ada Lovelace (sm_89) 内核与 Driver 590.48.01 (CUDA 13.1) 不兼容
    ↓
Conv1d 调用 cuDNN → PTX JIT 编译出非法指令 → GPU 执行 → Segfault
```

---

## 4. 修复步骤

```bash
# 1. 清 pip 缓存 + 重装真正的 cuDNN 8.7.0
pip cache purge
pip install --no-cache-dir --force-reinstall nvidia-cudnn-cu11==8.7.0.84

# 2. 验证
python -c "import torch; print(torch.backends.cudnn.version())"  # → 8700

# 3. 删除系统 cuDNN（可选，防御性清理）
sudo rm /usr/local/cuda-11.8/lib64/libcudnn*
sudo rm /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn*
sudo rm -rf /usr/local/cuda-11.8/include/cudnn*
sudo ldconfig

# 4. 验证训练
bash script_me/main_segmentation_train.sh
```

---

## 5. 经验教训

### 5.1 排查思路

> **遇到段错误：先做最小复现，逐层隔离变量。**
>
> 不要一上来就怀疑配置、数据、模型——用一个最简单的 `torch.randn + Conv1d` 就能定位。

### 5.2 cuDNN 版本来源三路径

| 路径 | 优先级 | 如何检查 |
|------|-------|---------|
| `LD_LIBRARY_PATH` | 最高 | `echo $LD_LIBRARY_PATH` |
| `ldconfig` 缓存 | 次高 | `ldconfig -p \| grep cudnn` |
| conda/pip 自带 (RPATH) | 最低 | `LD_DEBUG=libs python ...` |

三者中任何一个指向 8.9.7 都会导致段错误。

### 5.3 PIP 缓存风险

`pip cache purge` 清出了 31GB 缓存。旧缓存可能包含被污染/替换过的 wheel 文件，安装时不会重新校验完整性。遇到类似"重装后还是旧版本"的问题，**必须** `--no-cache-dir`。

### 5.4 系统 CUDA 与 conda 的关系

```
conda 环境自带完整的 CUDA 运行时 + cuDNN（pip install torch 自动拉取）
系统 /usr/local/cuda-* 通常不需要。——除非编译 CUDA 扩展时需要 nvcc
```

### 5.5 NVIDIA 驱动跨大版本兼容性

| 驱动 CUDA | cuDNN/PyTorch CUDA | 兼容 |
|----------|-------------------|------|
| 13.1 | 11.8 (cu118) | ⚠️ 大部分 OK（非 cuDNN 操作正常） |
| 13.1 | 11.8 + cuDNN 8.9.7 | ❌ Segfault（PTX 不兼容） |
| 13.1 | 11.8 + cuDNN 8.7.0 | ✅ 正常（无 sm_89 优化路径） |
| 13.1 | 12.1+ (cu121) | ✅ 推荐升级目标 |

### 5.6 cuDNN 版本速查

| 版本号 | `torch.backends.cudnn.version()` | 说明 |
|--------|-------------------------------|------|
| 8.7.0 | 8700 | ✅ 稳定 |
| 8.9.7 | 8907 | ❌ 此环境不支持 |
