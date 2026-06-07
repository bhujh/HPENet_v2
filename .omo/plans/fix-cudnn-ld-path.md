# 卸载系统 cuDNN 8.9.7 + 修改 LD_LIBRARY_PATH

## 根因

`~/.bashrc` 第 158 行的 `LD_LIBRARY_PATH` 让系统 cuDNN 8.9.7 抢在 conda 自带 cuDNN 8.7.0 之前加载，8.9.7 与 NVIDIA 驱动 590.48.01 不兼容导致段错误。

## 执行

### 1. 卸载系统 cuDNN 8.9.7

```bash
# 先看删什么
find /usr/local/cuda-11.8/ -name "libcudnn*" -type f 2>/dev/null

# 创建回滚日志
find /usr/local/cuda-11.8/ -name "libcudnn*" -type f 2>/dev/null | while read f; do
  echo "$f  $(sha256sum "$f" | cut -d' ' -f1)"
done > ~/cuDNN_rollback_$(date +%Y%m%d).log

# 删除 8.9.7（lib64 下）
sudo rm /usr/local/cuda-11.8/lib64/libcudnn*.8.9.7

# 检查 targets 下是否也有
ls /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn*.8.9.7 2>/dev/null && \
  sudo rm /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn*.8.9.7

# 删除断开的 symlink（如果 libcudnn.so.8 指向已删文件）
[ -L /usr/local/cuda-11.8/lib64/libcudnn.so.8 ] && [ ! -e /usr/local/cuda-11.8/lib64/libcudnn.so.8 ] && \
  sudo rm /usr/local/cuda-11.8/lib64/libcudnn.so.8 /usr/local/cuda-11.8/lib64/libcudnn.so

# 刷新动态链接缓存
sudo ldconfig
```

### 2. 修改 .bashrc

```bash
# 注释第 158 行
sed -i '158s/^export LD_LIBRARY_PATH/# export LD_LIBRARY_PATH/' ~/.bashrc

# 验证
bash -l -c 'echo $LD_LIBRARY_PATH | grep cuda-11.8 || echo "已干净"'
```

## 验证

```bash
conda activate hpenet
python -c "import torch; print(torch.backends.cudnn.version())"  # 应输出 8700
bash script_me/main_segmentation_train.sh                         # 应无 Segfault
```

## Must NOT

- ❌ 不删 `/usr/local/cuda-11.8/lib64/` 下的非 cuDNN 文件
- ❌ 不删 cuDNN 8.7.0（系统自带的旧版本）
- ❌ 不改 .bashrc 其他行
