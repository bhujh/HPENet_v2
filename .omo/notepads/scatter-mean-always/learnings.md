# scatter-mean-always 落地证据

> 方案：注释 main.py L651 修复 nearest_neighbor 误判（voxel_size=0.02 下 count.max()==1 文件误判 → L706 None 索引 CUDA 越界）
> 落地时间：2026-08-26

## 改动

`examples/segmentation/main.py` L651（唯一改动，1 行注释）：
改前 `        nearest_neighbor = len_part == 1`
改后 `        # nearest_neighbor = len_part == 1  # 误判源：...注释后 nearest_neighbor 回到 L641 配置驱动。`

## 验证结果

- 验证 1（grep 位置）：nearest_neighbor 共 5 处——L124（load_data 字符串比较）、L641（配置驱动赋值，保留）、L651（已注释）、L655/L700（引用）。
- 语法检查：ast.parse 通过。
- 前提复检：grep -rn test_mode script_me/ cfgs/radar/ 无输出（默认 multi_voxel 确认）。
- 验证 2（58 文件测试集）：CUDA_VISIBLE_DEVICES=1 ... mode=test --pretrained_path log/radar/radar-train-hpenet-ll-ngpus1-20260825-161134-bCqfckeRahnv9BdexbD6ni/checkpoint/..._ckpt_best.pth
  - 日志 results/scatter-mean-always-test.log
  - 无 CUDA 越界；58 文件全部跑通
  - 整体 test_oa 92.65 / test_macc 90.13 / test_miou 78.67
  - 量级 sanity：test_miou 78.67 vs 训练 val_miou 79.47（差 0.8pp）
- 验证 3（回归抽查）：per-file test_oa 全在 86.40~96.91，无归零点。

## 结论

修复生效，58 文件全部跑通，精度与训练 val 量级一致。
