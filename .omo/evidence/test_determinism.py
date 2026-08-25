"""Task 2: 确定性验证 - restore seed(100) 后两次运行一致性"""
import numpy as np
import sys, os, pickle, hashlib
sys.path.insert(0, '/home/wangpeng/CODE/HPENet_v2-main')
os.chdir('/home/wangpeng/CODE/HPENet_v2-main')

# 使用 seed(100) 确保确定性
np.random.seed(100)

# 构造合成点云 (X,Y,Z + 2 个特征)
N = 100
coord = np.random.randn(N, 3).astype(np.float32) * 5.0
feat = np.random.randn(N, 2).astype(np.float32)

from deploy.common import preprocess_test

run1 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3)
run2 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3)

# 对比 idx_points (子云索引列表)
same = True
for i, (r1, r2) in enumerate(zip(run1[2], run2[2])):
    if not np.array_equal(r1, r2):
        same = False
        print(f"DIFF at sub-cloud {i}: len={len(r1)} vs {len(r2)}")
        break

print(f"两次运行子云索引一致: {same}")
print(f"子云数量: {len(run1[2])}")
print(f"总点数: {N}")

# 哈希
h1 = hashlib.sha256(pickle.dumps(run1[2])).hexdigest()[:12]
h2 = hashlib.sha256(pickle.dumps(run2[2])).hexdigest()[:12]
print(f"run1 sha256: {h1}")
print(f"run2 sha256: {h2}")

assert same, "DETERMINISM FAILED!"
print("PASS: seed(100) ensures deterministic preprocess_test output")
