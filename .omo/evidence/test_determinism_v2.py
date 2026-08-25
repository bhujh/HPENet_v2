"""Task 2: 确定性验证 - 每文件独立 seed 模型"""
import numpy as np
import sys, os, hashlib
sys.path.insert(0, '/home/wangpeng/CODE/HPENet_v2-main')
os.chdir('/home/wangpeng/CODE/HPENet_v2-main')

N = 100
coord = np.random.RandomState(42).randn(N, 3).astype(np.float32) * 5.0
feat = np.random.RandomState(42).randn(N, 2).astype(np.float32)

from deploy.common import preprocess_test

# 场景 A: 全局 seed(100) 一次 + 两次调用（模拟当前行为）
np.random.seed(100)
r1 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3)
np.random.seed(100)  # 重新 reseed
r2 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3)
same_per_file = all(np.array_equal(a, b) for a, b in zip(r1[2], r2[2]))
print(f"[每文件独立 seed] 两次调用一致: {same_per_file}")

# 场景 B: 全局 seed 仅一次（模拟修复后的 trt_inference.py 实际行为）
np.random.seed(100)
r3 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3)
r4 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3)  # 不 reseed
same_global = all(np.array_equal(a, b) for a, b in zip(r3[2], r4[2]))
print(f"[全局连续状态] 两次调用一致: {same_global}")

# 场景 C: 局部 RandomState(100) - C++ NumpyMT19937 per-file 模型
rng = np.random.RandomState(100)
# 模拟 preprocess_test 用 rng.shuffle 替代 np.random.shuffle
# (当前 common.py 用 np.random.shuffle，无法接外部 rng)
print(f"\n[C++ 等效模型] 每文件独立 seed(100) → per-file 确定性")

print(f"\n结论:")
print(f"  per-file seed: {same_per_file} (需要每文件 reseed 或传局部 RandomState)")
print(f"  global state:  {same_global} (当前修复后仍不一致！)")

# 记录
with open('.omo/evidence/task-2-determinism.txt', 'w') as f:
    f.write(f"per_file_same: {same_per_file}\nglobal_same: {same_global}\n")
    f.write("结论: 恢复全局 seed(100) 不足以保证跨文件确定性\n")
    f.write("方案: preprocess_test 需支持显式 seed 参数或接受 RandomState 实例\n")
